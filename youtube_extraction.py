from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
import time
import requests
from check_validity import isValid

# Library: https://pytubefix.readthedocs.io/en/latest/
url = "https://www.youtube.com/watch?v=jvXfxs0bTho"
path = "Videos"
def download(url,path):
    yt = YouTube(url, on_progress_callback=on_progress)

    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=path)
download(url=url,path=path)

# ==========================
# CONFIG
# ==========================

# API info @ https://console.cloud.google.com/apis/credentials?project=erging-video-retrieval

API_KEY = "AIzaSyAW1yMAXpBdtvYdH9Xs9yioA18EIzbJnVE"

SEARCH_QUERIES = [
    "2k erg test",
    "rowing 2k erg",
    "concept2 erg test",
    "erg test rowing",
    "6k erg",
]

MAX_RESULTS_PER_QUERY = 10
DOWNLOAD_DIR = "Videos"


KEYWORDS_MUST_HAVE = [
    "erg",
    "rowing",
    "row",
    "concept2",
]

KEYWORDS_EXCLUDE = [
    "ski erg",
    "bikeerg",
    "bike erg",
]


# ==========================
# YOUTUBE API HELPERS
# ==========================
def search_youtube(query, api_key, max_results=20):
    """Search YouTube for a query and return raw search results."""
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": api_key,
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "videoDuration": "medium",  # ~4–20 mins; good range for 2k/6k tests
        "safeSearch": "none",
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("items", [])

# TODO: Skip to middle of video and use pose detection confidence to determine if video has side-profile erging
# TODO: Account for different camera angles

def is_relevant(snippet):
    """Heuristic filter: keep videos whose title/description look erg-related."""
    title = snippet.get("title", "")
    desc = snippet.get("description", "")
    text = (title + " " + desc).lower()

    # Must have at least one positive keyword
    if not any(kw in text for kw in KEYWORDS_MUST_HAVE):
        return False

    # Must not contain excluded words
    if any(bad in text for bad in KEYWORDS_EXCLUDE):
        return False

    return True


def collect_video_ids():
    """Search all queries and return a de-duplicated list of relevant video IDs."""
    video_ids = set()

    for q in SEARCH_QUERIES:
        print(f"\n[SEARCH] Query: {q}")
        try:
            items = search_youtube(q, API_KEY, max_results=MAX_RESULTS_PER_QUERY)
        except requests.HTTPError as e:
            print(f"  ! HTTP error during search: {e}")
            continue

        for item in items:
            snippet = item.get("snippet", {})
            if not is_relevant(snippet):
                continue

            vid_id = item["id"]["videoId"]
            title = snippet.get("title", "Untitled")
            if vid_id not in video_ids:
                video_ids.add(vid_id)
                print(f"  + Found relevant video: {title} (id={vid_id})")

        # tiny delay to be nice to the API
        time.sleep(0.3)

    print(f"\n[INFO] Total unique relevant videos: {len(video_ids)}")
    return list(video_ids)

print(search_youtube(query = SEARCH_QUERIES[:5], api_key=API_KEY, max_results=MAX_RESULTS_PER_QUERY))