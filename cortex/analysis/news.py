"""News and social (X) sentiment aggregator.

Sources are intentionally pluggable. By default Cortex uses free public
RSS feeds (so it works without any credentials) plus a deterministic
sentiment scorer. When an LLM key is configured, sentiment is upgraded
with a deep reasoning pass.

Notes on sources:
  * Alpaca's REST news endpoint requires a non-paper data subscription,
    so it is only attempted when explicitly enabled.
  * X (Twitter) social traffic can be added by supplying an endpoint that
    returns JSON posts; see `X_SOCIAL_ENDPOINTS`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests
import xml.etree.ElementTree as ET

from config.settings import settings
from cortex.clients.llm_client import LLMClient, sentiment_from_text

log = logging.getLogger(__name__)

# Free, no-key market news RSS feeds (diverse sources beyond Yahoo/CNBC).
RSS_FEEDS = [
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.reutersagency.com/feed/?best-topics=markets&post_type=best",
    "https://finance.yahoo.com/rss/topstories",
]

# Pluggable endpoint returning {"posts": [{"text": "...", "source": "..."}]}.
X_SOCIAL_ENDPOINTS = [
    # Add a URL here if you have an X/social data source you want Cortex to read.
]


@dataclass
class Headline:
    title: str
    link: str
    source: str
    published: str = ""
    sentiment: str = "neutral"
    sentiment_score: float = 0.0

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "link": self.link,
            "source": self.source,
            "published": self.published,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
        }


class NewsAggregator:
    def __init__(self, llm: Optional[LLMClient] = None, max_items: int = 25) -> None:
        self.llm = llm
        self.max_items = max_items

    # -------- RSS fetch (free, no key) --------
    def _fetch_rss(self, url: str) -> list[dict]:
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "CortexTradingBot/0.1"})
            resp.raise_for_status()
        except Exception as exc:
            log.debug("RSS fetch failed %s: %s", url, exc)
            return []
        root = ET.fromstring(resp.text)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub = item.findtext("pubDate") or ""
            if title:
                items.append({"title": title, "link": link, "published": pub})
        return items

    # -------- public sources --------
    def fetch_market_news(self) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for feed in RSS_FEEDS:
            for it in self._fetch_rss(feed):
                t = it["title"].strip()
                if t in seen:
                    continue
                seen.add(t)
                it["source"] = _source_from_feed(feed)
                out.append(it)
            if len(out) >= self.max_items + 20:
                break
        return out[: self.max_items]

    # -------- social (X) --------
    def fetch_social(self) -> list[str]:
        posts: list[str] = []
        for endpoint in X_SOCIAL_ENDPOINTS:
            try:
                resp = requests.get(endpoint, timeout=20)
                data = resp.json()
                posts += [p.get("text", "") for p in data.get("posts", []) if p.get("text")]
            except Exception as exc:
                log.debug("Social fetch failed %s: %s", endpoint, exc)
        return posts

    # -------- sentiment & enrichment --------
    def enrich(self, items: list[dict]) -> list[Headline]:
        headlines = []
        for it in items[: self.max_items]:
            text = it["title"]
            sent = sentiment_from_text(text)
            headlines.append(
                Headline(
                    title=text,
                    link=it.get("link", ""),
                    source=it.get("source", "unknown"),
                    published=it.get("published", ""),
                    sentiment=sent["label"],
                    sentiment_score=sent["score"],
                )
            )
        return headlines

    def analyze_social_sentiment(self, posts: list[str]) -> dict:
        """Aggregate sentiment over social posts (rules-based, upgradeable via LLM)."""
        if not posts:
            return {"count": 0, "label": "neutral", "score": 0.0, "summary": "No social posts available."}
        scores = [sentiment_from_text(p)["score"] for p in posts]
        labels = [sentiment_from_text(p)["label"] for p in posts]
        avg = sum(scores) / len(scores)
        label = "bullish" if avg > 0.15 else ("bearish" if avg < -0.15 else "neutral")
        summary = f"{len(posts)} posts parsed; avg score {avg:.2f} ({label})."
        if self.llm is not None and self.llm.enabled:
            try:
                sample = "\n".join(posts[:20])
                j = self.llm.ask_json(
                    f"Summarise overall market sentiment from these X/social posts "
                    f"and note any dominant tickers:\n{sample}",
                )
                summary = j.get("summary", summary)
            except Exception as exc:
                log.warning("LLM social analysis failed: %s", exc)
        return {"count": len(posts), "label": label, "score": round(avg, 3), "summary": summary}

    def get_news_brief(self) -> dict:
        raw = self.fetch_market_news()
        headlines = self.enrich(raw)
        social = self.fetch_social()
        social_sent = self.analyze_social_sentiment(social)
        return {"headlines": [h.as_dict() for h in headlines], "social": social_sent}

    def market_events_brief(self, headline_limit: int = 12) -> dict:
        """Up-to-date market events: headlines + an LLM-summarised brief.

        Returns the raw headlines plus a concise natural-language summary of
        the most important recent events affecting the market/tickers being
        watched, with sources attached. Falls back to a flat list when no LLM
        is configured.
        """
        raw = self.fetch_market_news()
        headlines = [h.as_dict() for h in self.enrich(raw[:headline_limit])]
        events: dict = {"headlines": headlines, "summary": ""}

        if self.llm is not None and self.llm.enabled and headlines:
            items = "\n".join(
                f"- [{h['source']}] {h['title']}" for h in headlines[:headline_limit]
            )
            try:
                j = self.llm.ask_json(
                    "Here are today's market headlines. Produce a concise 'recent events'\n"
                    "brief (max 6 bullets) summarising the most important developments and\n"
                    "their likely market impact, and list the key tickers affected.\n"
                    "Return JSON with keys 'summary' (string) and 'key_tickers' (list).\n\n"
                    f"{items}"
                )
                events["summary"] = j.get("summary", "")
                events["key_tickers"] = j.get("key_tickers", [])
            except Exception as exc:
                log.warning("LLM market-events brief failed: %s", exc)
                events["summary"] = "LLM brief unavailable; see headlines below."
        return events


def _source_from_feed(feed: str) -> str:
    if "mw_" in feed or "marketwatch" in feed:
        return "marketwatch"
    for name in ("bbc", "reuters", "yahoo"):
        if name in feed:
            return name
    return "rss"