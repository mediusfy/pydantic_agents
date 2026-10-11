from pydantic_agents.core.tools.registry import register_tool

try:
    from langchain_community.tools import DuckDuckGoSearchRun

    _duckduckgo = DuckDuckGoSearchRun()

    def search_web_duckduckgo(query: str) -> str:
        """Search the web using DuckDuckGo (via LangChain). Useful for finding recent information, docs, or code examples."""
        try:
            return _duckduckgo.run(query)
        except Exception as e:
            return f"Search failed: {e}"

    register_tool(
        "search_web_duckduckgo",
        search_web_duckduckgo,
        "Search the web using DuckDuckGo. Useful for finding recent information, documentation, or code examples.",
        ["web", "search", "research"],
    )
except ImportError:
    pass

try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    _wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    def search_wikipedia(query: str) -> str:
        """Search Wikipedia for information on a topic (via LangChain)."""
        try:
            return _wikipedia.run(query)
        except Exception as e:
            return f"Wikipedia search failed: {e}"

    register_tool(
        "search_wikipedia",
        search_wikipedia,
        "Search Wikipedia for information on a topic. Useful for general knowledge and definitions.",
        ["research", "docs", "web"],
    )
except ImportError:
    pass

try:
    from langchain_community.tools import ArxivQueryRun

    _arxiv = ArxivQueryRun()

    def search_arxiv(query: str) -> str:
        """Search arXiv for academic papers on a topic (via LangChain)."""
        try:
            return _arxiv.run(query)
        except Exception as e:
            return f"arXiv search failed: {e}"

    register_tool(
        "search_arxiv",
        search_arxiv,
        "Search arXiv for academic papers and research. Useful for technical research and citations.",
        ["research", "search"],
    )
except ImportError:
    pass
