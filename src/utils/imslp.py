"""Looks up and fetches imslp pdf score."""
import logging
import re
from asyncio import run
from pathlib import Path
from typing import Optional, cast
from urllib.parse import unquote

from aiohttp import ClientSession
from bs4 import BeautifulSoup, Tag

# https://duckduckgo.com/l/?uddg=https%3A%2F%2Fimslp.org%2Fwiki%2FInvention_in_D_minor%2C_BWV_775_(Bach%2C_Johann_Sebastian)&rut=d3f336a6dd9bbdc118e0f8e2a6f4d1eb03c0b44cf947937d048ea73da42ae682
# https://imslp.org/wiki/Invention_in_D_minor,_BWV_775_(Bach,_Johann_Sebastian)


class IMSLP:
    session: ClientSession

    IMSLP_BASE_URL = "https://imslp.org"

    URL_LINK_RE = re.compile(r'^.*\?uddg=([^\&]*)\&.*$')

    def __init__(self):
        self.session = ClientSession()

    async def find_imslp(self, query: str) -> Optional[str]:
        """Query DuckDuckGo for an IMSLP page.

        Args:
            query (str): The query to send Google.

        Returns:
            Optional[str]: The first imslp link in Google search results,
            or None if none was found.
        """
        async with self.session.get(
            url="https://www.duckduckgo.com/html",
            headers={
                "User-Agent": "Lynx/2.8.9rel.1 libwww-FM/2.14 SSL-MM/1.4.1 OpenSSL/1.0.0a",
                "Accept": "*/*"
            },
            params={
                "q": query + " site:imslp.org",
                "client": "ubuntu-chr",
                "sourceid": "chrome&ie=UTF-",
                "ie": "UTF-8",
            },
            cookies={
                'CONSENT': 'PENDING+987',  # Bypasses the consent page
                'SOCS': 'CAESHAgBEhIaAB',
            }
        ) as resp:
            resp.raise_for_status()
            text = await resp.text()
            soup = BeautifulSoup(text, "html.parser")
            for a in soup.find_all("a", {"class": "result__a"}):
                if (m := self.URL_LINK_RE.match(str(a.get("href", "")))):
                    url = unquote(m.group(1))
                    if url.startswith("https://imslp.org/"):
                        return url
        return None

    COMPLETE_SCORE_RE = re.compile(r'^.*Complete Score.*$')
    DOWNLOADS_RE = re.compile(r'^Total number of downloads:\s*([0-9]+)\s*$')

    def _extract_download_links(self, content: bytes) -> list[tuple[str, int]]:
        links: list[tuple[str, int]] = list([])
        soup = BeautifulSoup(content, "html.parser")
        scores = soup.find("div", {"id": "wpscore_tabs"})
        if scores is None:
            return links
        for file in scores.find_all("div", {"class": "we_file_first"}):
            # Find the download counter:
            span = file.find(
                "span",
                title=lambda t: t and self.DOWNLOADS_RE.match(
                    str(t))  # type: ignore
            )
            downloads: int = -1
            if span and (title := span["title"]) and (m := self.DOWNLOADS_RE.match(str(title))):
                downloads = int(m.group(1))
            if downloads < 0:
                continue
            # Find the download link.
            for a in file.find_all("a"):
                if a.has_attr("rel") and "nofollow" in a["rel"]:
                    span = a.find("span")
                    if span and span.has_attr('title') and span["title"] == "Download this file":
                        if self.COMPLETE_SCORE_RE.match(span.text):
                            # Got it: add to results,
                            links.append((str(a["href"]), downloads))
        return links

    # IMSLP_COOKIES = r'imslp_wikiLanguageSelectorLanguage=en; chatbase_anon_id=d8925c94-d976-492a-9649-e563f973d8a2; imslpdisclaimeraccepted=yes; __stripe_mid=5d13801d-837c-4919-8e35-88de460c440b313847; _gid=GA1.2.642930185.1737548859; __stripe_sid=d726e726-eeea-4292-b94d-715cac65d6979cf564; _ga_4QW4VCTZ4E=GS1.1.1737559129.13.1.1737560753.0.0.0; _ga=GA1.2.1606208118.1735899643; _ga_8370FT5CWW=GS1.2.1737559147.12.1.1737560755.0.0.0'
    IMSLP_COOKIES = {
        "imslp_wikiLanguageSelectorLanguage": "en",
        "chatbase_anon_id": "d8925c94-d976-492a-9649-e563f973d8a2",
        "imslpdisclaimeraccepted": "yes",
        "redirectPassed": "1"
    }

    async def find_pdf_links(self, imslp_page: str) -> list[tuple[str, int]]:
        """Extracts all download links from the given IMSLP page.

        Args:
            imslp_page (str): URL of the IMSLP page.

        Returns:
            list[tuple[str, int]]: List of links and their download counter.
        """
        logging.info(f"Extracting download links from {imslp_page}")
        # Fetches the page and find the download link.
        async with self.session.get(imslp_page) as resp:
            resp.raise_for_status()
            content = await resp.read()
            links = self._extract_download_links(content)
            links.sort(key=lambda t: t[1], reverse=True)
            return links

    ANCHOR_TEXT = "I agree with the disclaimer above, continue my download"

    async def download_link(self, pdf_link) -> Optional[str]:
        async with self.session.post(
            url=pdf_link,
            headers={
                "User-Agent": "Lynx/2.8.9rel.1 libwww-FM/2.14 SSL-MM/1.4.1 OpenSSL/1.0.0a",
                "Accept": "*/*"
            },
            cookies=self.IMSLP_COOKIES
        ) as resp:
            resp.raise_for_status()
            content = await resp.read()
            soup = BeautifulSoup(content, "html.parser")
            span = cast(Tag, soup.find("span", id="sm_dl_wait"))
            a = soup.find("a", string=self.ANCHOR_TEXT)  # type:ignore
            if span and span.has_attr('data-id'):
                return cast(str, span['data-id'])
            elif a and a.has_attr("href"):
                return "https://imslp.eu" + str(a['href'])
            return None

    async def save_pdf(self, download_url: str, into: Path):
        logging.info(f"\tSaving {download_url} to {into}")
        async with self.session.get(download_url) as resp:
            content = await resp.read()
            with open(into, "wb+") as fp:
                fp.write(content)
            return True

        return False


async def async_run():
    imslp = IMSLP()
    # imslp_page = await imslp.find_imslp("bach invention no 1")
    imslp_page = "https://imslp.org/wiki/Invention_in_C_major,_BWV_772_(Bach,_Johann_Sebastian)"
    if imslp_page is None:
        return None

    print(f"Fetching {imslp_page} for pdf links...")
    links = await imslp.find_pdf_links(imslp_page)
    print(f"Found imslp links: ")
    for (url, count) in links:
        print(f"\t{count}: {url}")
    selected_link, _ = links[0]
    pdf_link = await imslp.download_link(selected_link)
    if pdf_link is not None:
        print(f"Saving {pdf_link}")
        await imslp.save_pdf(pdf_link, Path("anselm.pdf"))
    else:
        print("Not found")


def main():
    run(async_run())


if __name__ == "__main__":
    main()

# vscode - End of file
