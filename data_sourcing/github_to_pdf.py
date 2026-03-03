"""
GitHub profile → PDF generator.

Fetches public GitHub data for a user and produces a clean PDF document
suitable for ingestion into the RAG pipeline.

Usage:
    python -m data_sourcing.github_to_pdf
    # or with a token for pinned repos + higher rate limit:
    GITHUB_TOKEN=ghp_... python -m data_sourcing.github_to_pdf
"""

from __future__ import annotations

import os
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

GITHUB_API = "https://api.github.com"
GRAPHQL_API = "https://api.github.com/graphql"

DEFAULT_USERNAME = "aashishravindran"
DEFAULT_OUTPUT = "data_sourcing/data/github_profile.pdf"

# PDF colour palette
COLOR_HEADER_BG = (30, 58, 138)   # deep blue
COLOR_HEADER_FG = (255, 255, 255)  # white
COLOR_BODY = (30, 30, 30)          # near-black
COLOR_SECONDARY = (100, 100, 100)  # medium grey
COLOR_BAR_FILL = (59, 130, 246)    # blue
COLOR_BAR_BG = (220, 220, 220)     # light grey
COLOR_SEP = (200, 200, 200)        # separator


# ── Exceptions ─────────────────────────────────────────────────────────────────

class GitHubAPIError(Exception):
    def __init__(self, message: str, status_code: int = 0, endpoint: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint


class GitHubRateLimitError(GitHubAPIError):
    pass


class GitHubNotFoundError(GitHubAPIError):
    pass


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UserProfile:
    name: str
    bio: str
    blog: str
    location: str
    company: str
    followers: int
    following: int
    public_repos: int
    member_since: int  # year


@dataclass(frozen=True)
class RepoInfo:
    name: str
    description: str
    language: str
    stars: int
    forks: int
    topics: tuple[str, ...]
    html_url: str
    pushed_at: str  # ISO 8601


@dataclass
class ContributionStats:
    total_pushes: int = 0
    total_prs: int = 0
    repos_touched: int = 0
    most_active_month: str = "N/A"
    event_type_counts: Counter = field(default_factory=Counter)


# ── GitHub Client ──────────────────────────────────────────────────────────────

class GitHubClient:
    def __init__(self, username: str, token: Optional[str] = None):
        self.username = username
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.github+json"
        self.session.headers["X-GitHub-Api-Version"] = "2022-11-28"
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self._token = token

    def _get(self, path: str, params: Optional[dict] = None) -> dict | list:
        url = f"{GITHUB_API}{path}"
        response = self.session.get(url, params=params)

        remaining = int(response.headers.get("X-RateLimit-Remaining", 999))
        if remaining < 5:
            reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))
            reset_time = datetime.fromtimestamp(reset_ts).strftime("%H:%M:%S")
            warnings.warn(
                f"GitHub API rate limit nearly exhausted ({remaining} remaining). "
                f"Resets at {reset_time}."
            )

        if response.status_code == 403 and "rate limit" in response.text.lower():
            raise GitHubRateLimitError(
                "GitHub API rate limit exceeded. Set GITHUB_TOKEN to increase limit.",
                status_code=403,
                endpoint=url,
            )
        if response.status_code == 404:
            raise GitHubNotFoundError(
                f"Resource not found: {url}", status_code=404, endpoint=url
            )
        if not response.ok:
            raise GitHubAPIError(
                f"GitHub API error {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
                endpoint=url,
            )
        return response.json()

    def get_profile(self) -> UserProfile:
        data = self._get(f"/users/{self.username}")
        return UserProfile(
            name=data.get("name") or self.username,
            bio=data.get("bio") or "",
            blog=data.get("blog") or "",
            location=data.get("location") or "",
            company=data.get("company") or "",
            followers=data.get("followers", 0),
            following=data.get("following", 0),
            public_repos=data.get("public_repos", 0),
            member_since=int(data["created_at"][:4]),
        )

    def get_repos(self) -> list[RepoInfo]:
        data = self._get(
            f"/users/{self.username}/repos",
            params={"per_page": 100, "sort": "pushed", "direction": "desc"},
        )
        repos = []
        for r in data:
            if r.get("fork"):
                continue
            repos.append(
                RepoInfo(
                    name=r["name"],
                    description=r.get("description") or "",
                    language=r.get("language") or "",
                    stars=r.get("stargazers_count", 0),
                    forks=r.get("forks_count", 0),
                    topics=tuple(r.get("topics", [])),
                    html_url=r["html_url"],
                    pushed_at=r.get("pushed_at", ""),
                )
            )
        return repos

    def get_pinned_repos(self, all_repos: list[RepoInfo]) -> list[RepoInfo]:
        """Returns up to 6 pinned repos via GraphQL, or falls back to top-6 by pushed_at."""
        if self._token:
            query = """
            query {
              user(login: "%s") {
                pinnedItems(first: 6, types: REPOSITORY) {
                  nodes {
                    ... on Repository {
                      name
                      description
                      primaryLanguage { name }
                      stargazerCount
                      forkCount
                      repositoryTopics(first: 5) {
                        nodes { topic { name } }
                      }
                      url
                      pushedAt
                    }
                  }
                }
              }
            }
            """ % self.username
            resp = self.session.post(GRAPHQL_API, json={"query": query})
            if resp.ok:
                try:
                    nodes = (
                        resp.json()["data"]["user"]["pinnedItems"]["nodes"]
                    )
                    pinned = []
                    for n in nodes:
                        topics = tuple(
                            t["topic"]["name"]
                            for t in n.get("repositoryTopics", {}).get("nodes", [])
                        )
                        lang = ""
                        if n.get("primaryLanguage"):
                            lang = n["primaryLanguage"]["name"]
                        pinned.append(
                            RepoInfo(
                                name=n["name"],
                                description=n.get("description") or "",
                                language=lang,
                                stars=n.get("stargazerCount", 0),
                                forks=n.get("forkCount", 0),
                                topics=topics,
                                html_url=n.get("url", ""),
                                pushed_at=n.get("pushedAt", ""),
                            )
                        )
                    if pinned:
                        return pinned
                except (KeyError, TypeError):
                    pass
            warnings.warn("GraphQL pinned repos failed; falling back to top-6 by activity.")

        # Fallback: top-6 non-fork repos already sorted by pushed_at
        return all_repos[:6]

    def get_language_breakdown(self, repos: list[RepoInfo]) -> dict[str, int]:
        aggregate: dict[str, int] = {}
        for repo in repos[:15]:  # cap to stay within rate limit budget
            try:
                lang_data = self._get(f"/repos/{self.username}/{repo.name}/languages")
                for lang, bytes_count in lang_data.items():
                    aggregate[lang] = aggregate.get(lang, 0) + bytes_count
            except GitHubAPIError:
                continue
        return dict(sorted(aggregate.items(), key=lambda x: x[1], reverse=True))

    def get_contribution_stats(self) -> ContributionStats:
        stats = ContributionStats()
        repos_touched: set[str] = set()
        month_counter: Counter = Counter()

        for page in range(1, 4):  # max 300 events (GitHub hard limit)
            try:
                events = self._get(
                    f"/users/{self.username}/events",
                    params={"per_page": 100, "page": page},
                )
            except GitHubAPIError:
                break
            if not events:
                break

            for event in events:
                etype = event.get("type", "")
                stats.event_type_counts[etype] += 1
                repo_name = event.get("repo", {}).get("name", "")
                if repo_name:
                    repos_touched.add(repo_name)
                created = event.get("created_at", "")
                if created:
                    month_counter[created[:7]] += 1  # "YYYY-MM"
                if etype == "PushEvent":
                    stats.total_pushes += 1
                elif etype == "PullRequestEvent":
                    stats.total_prs += 1

        stats.repos_touched = len(repos_touched)
        if month_counter:
            top_month = month_counter.most_common(1)[0][0]
            dt = datetime.strptime(top_month, "%Y-%m")
            stats.most_active_month = dt.strftime("%B %Y")
        return stats


# ── PDF Generator ──────────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """Replace non-Latin-1 characters to avoid fpdf2 encoding errors."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _pct_bar(pdf: FPDF, label: str, pct: float, bar_width: float = 80.0) -> None:
    """Render a single text-based percentage bar row."""
    fill_w = bar_width * pct / 100
    empty_w = bar_width - fill_w

    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*COLOR_BODY)
    pdf.cell(w=45, h=6, text=_sanitize(label))

    # filled portion
    pdf.set_fill_color(*COLOR_BAR_FILL)
    if fill_w > 0:
        pdf.cell(w=fill_w, h=5, text="", fill=True)
    # empty portion
    pdf.set_fill_color(*COLOR_BAR_BG)
    if empty_w > 0:
        pdf.cell(w=empty_w, h=5, text="", fill=True)

    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*COLOR_SECONDARY)
    pdf.cell(w=15, h=5, text=f"  {pct:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _section_header(pdf: FPDF, title: str) -> None:
    pdf.set_fill_color(*COLOR_HEADER_BG)
    pdf.set_text_color(*COLOR_HEADER_FG)
    pdf.set_font("Helvetica", style="B", size=10)
    pdf.cell(w=0, h=7, text=f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_text_color(*COLOR_BODY)


def _separator(pdf: FPDF) -> None:
    pdf.set_draw_color(*COLOR_SEP)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)


class GitHubProfilePDFGenerator:
    def __init__(self, client: GitHubClient, output_path: str = DEFAULT_OUTPUT):
        self.client = client
        self.output_path = output_path

    def generate(self) -> None:
        print(f"Fetching GitHub profile for @{self.client.username}...")
        profile = self.client.get_profile()
        print("  ✓ Profile")
        repos = self.client.get_repos()
        print(f"  ✓ Repos ({len(repos)} non-fork)")
        featured = self.client.get_pinned_repos(repos)
        print(f"  ✓ Featured repos ({len(featured)})")
        lang_breakdown = self.client.get_language_breakdown(repos)
        print(f"  ✓ Language breakdown ({len(lang_breakdown)} languages)")
        contrib = self.client.get_contribution_stats()
        print("  ✓ Contribution stats")

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_margins(left=15, top=12, right=15)
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        self._render_header(pdf, profile)
        _separator(pdf)
        _section_header(pdf, "PROFILE OVERVIEW")
        self._render_profile_overview(pdf, profile)
        _separator(pdf)
        _section_header(pdf, "FEATURED REPOSITORIES")
        self._render_featured_repos(pdf, featured)
        _separator(pdf)
        _section_header(pdf, "LANGUAGE BREAKDOWN")
        self._render_language_breakdown(pdf, lang_breakdown)
        _separator(pdf)
        _section_header(pdf, "RECENT ACTIVITY (last 300 public events)")
        self._render_contribution_stats(pdf, contrib)

        # Footer
        pdf.set_y(-15)
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(*COLOR_SECONDARY)
        pdf.cell(
            w=0, h=5,
            text=f"Generated: {datetime.now().strftime('%Y-%m-%d')}  |  "
                 f"github.com/{self.client.username}",
            align="C",
        )

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        pdf.output(self.output_path)
        print(f"\nPDF saved to: {self.output_path}")

    def _render_header(self, pdf: FPDF, profile: UserProfile) -> None:
        pdf.set_font("Helvetica", style="B", size=20)
        pdf.set_text_color(*COLOR_BODY)
        pdf.cell(w=0, h=10, text=_sanitize(profile.name), new_x="LMARGIN", new_y="NEXT")

        if profile.bio:
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(*COLOR_SECONDARY)
            pdf.multi_cell(w=0, h=6, text=_sanitize(profile.bio))
            pdf.ln(1)

        links = []
        if profile.blog:
            links.append(profile.blog)
        links.append(f"github.com/{self.client.username}")
        pdf.set_font("Helvetica", size=9)
        pdf.set_text_color(*COLOR_SECONDARY)
        pdf.cell(w=0, h=5, text=_sanitize("  •  ".join(links)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    def _render_profile_overview(self, pdf: FPDF, profile: UserProfile) -> None:
        fields = [
            ("Member Since", str(profile.member_since)),
            ("Public Repos", str(profile.public_repos)),
            ("Followers", str(profile.followers)),
            ("Following", str(profile.following)),
        ]
        if profile.location:
            fields.append(("Location", profile.location))
        if profile.company:
            fields.append(("Company", profile.company))

        for label, value in fields:
            pdf.set_font("Helvetica", style="B", size=9)
            pdf.set_text_color(*COLOR_BODY)
            pdf.cell(w=40, h=6, text=_sanitize(label + ":"))
            pdf.set_font("Helvetica", size=9)
            pdf.cell(w=0, h=6, text=_sanitize(value), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    def _render_featured_repos(self, pdf: FPDF, repos: list[RepoInfo]) -> None:
        for i, repo in enumerate(repos):
            # Repo name + language badge
            pdf.set_font("Helvetica", style="B", size=10)
            pdf.set_text_color(*COLOR_BODY)
            name_text = _sanitize(repo.name)
            if repo.language:
                name_text += f"  [{_sanitize(repo.language)}]"
            pdf.cell(w=130, h=6, text=name_text)

            pdf.set_font("Helvetica", size=9)
            pdf.set_text_color(*COLOR_SECONDARY)
            pdf.cell(
                w=0, h=6,
                text=f"* {repo.stars}   fork {repo.forks}",
                new_x="LMARGIN", new_y="NEXT",
            )

            if repo.description:
                pdf.set_font("Helvetica", style="I", size=9)
                pdf.set_text_color(*COLOR_SECONDARY)
                pdf.multi_cell(w=0, h=5, text=_sanitize(repo.description))

            if repo.topics:
                pdf.set_font("Helvetica", size=8)
                pdf.set_text_color(*COLOR_SECONDARY)
                pdf.cell(
                    w=0, h=5,
                    text="Topics: " + _sanitize(", ".join(repo.topics)),
                    new_x="LMARGIN", new_y="NEXT",
                )

            pushed = ""
            if repo.pushed_at:
                try:
                    dt = datetime.fromisoformat(repo.pushed_at.replace("Z", "+00:00"))
                    pushed = dt.strftime("%Y-%m-%d")
                except ValueError:
                    pushed = repo.pushed_at[:10]

            pdf.set_font("Helvetica", size=8)
            pdf.set_text_color(*COLOR_SECONDARY)
            pdf.cell(
                w=0, h=5,
                text=_sanitize(f"Last pushed: {pushed}  •  {repo.html_url}"),
                new_x="LMARGIN", new_y="NEXT",
            )

            if i < len(repos) - 1:
                pdf.ln(2)
                pdf.set_draw_color(*COLOR_SEP)
                pdf.line(pdf.l_margin + 10, pdf.get_y(), pdf.w - pdf.r_margin - 10, pdf.get_y())
                pdf.ln(3)

        pdf.ln(4)

    def _render_language_breakdown(self, pdf: FPDF, langs: dict[str, int]) -> None:
        total = sum(langs.values()) or 1
        for lang, bytes_count in list(langs.items())[:10]:
            pct = bytes_count / total * 100
            _pct_bar(pdf, lang, pct)
        pdf.ln(2)

    def _render_contribution_stats(self, pdf: FPDF, stats: ContributionStats) -> None:
        rows = [
            ("Push Events (commit proxy)", str(stats.total_pushes)),
            ("Pull Requests", str(stats.total_prs)),
            ("Repos Contributed To", str(stats.repos_touched)),
            ("Most Active Month", stats.most_active_month),
        ]
        for label, value in rows:
            pdf.set_font("Helvetica", style="B", size=9)
            pdf.set_text_color(*COLOR_BODY)
            pdf.cell(w=60, h=6, text=_sanitize(label + ":"))
            pdf.set_font("Helvetica", size=9)
            pdf.cell(w=0, h=6, text=_sanitize(value), new_x="LMARGIN", new_y="NEXT")

        pdf.ln(2)
        pdf.set_font("Helvetica", style="I", size=8)
        pdf.set_text_color(*COLOR_SECONDARY)
        pdf.cell(
            w=0, h=5,
            text="Note: based on public GitHub events API (max 300 events)",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.ln(4)


# ── Public Entry Point ─────────────────────────────────────────────────────────

def generate_github_pdf(
    username: str = DEFAULT_USERNAME,
    output_path: str = DEFAULT_OUTPUT,
    token: Optional[str] = None,
) -> None:
    if token is None:
        token = os.getenv("GITHUB_TOKEN")
    client = GitHubClient(username=username, token=token)
    generator = GitHubProfilePDFGenerator(client=client, output_path=output_path)
    generator.generate()


if __name__ == "__main__":
    generate_github_pdf()
