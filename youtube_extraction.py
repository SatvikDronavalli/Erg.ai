from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
import time
import requests

# Library: https://pytubefix.readthedocs.io/en/latest/
url = "https://www.youtube.com/watch?v=I40z-S4olHA&list=PL2lrNgnb31592s8xGwMd0JQzShjgE6bVV&index=4"
path = "test_dir"
os.makedirs(path, exist_ok=True)
def download(url,path, id):
    yt = YouTube(url, on_progress_callback=on_progress)
    ys = yt.streams.get_highest_resolution()
    ys.download(output_path=path)
download(url=url,path=path,id="test_vid_2")

# ==========================
# CONFIG
# ==========================

# API key @ https://console.cloud.google.com/apis/credentials?project=erging-video-retrieval
'''
API_KEY = os.environ.get("YOUTUBE_API_KEY")

SEARCH_QUERIES = [
    "2k erg test",
    "rowing 2k erg",
    "concept2 erg test",
    "erg test rowing",
    "6k erg",
]

MAX_RESULTS_PER_QUERY = 1
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


def collect_videos():
    """Search all queries and return a de-duplicated dictionary of relevant videos.
        The keys are the video id's and the values are the video titles"""
    videos = {}
    for q in SEARCH_QUERIES:
        # print(f"\n[SEARCH] Query: {q}")
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
            videos[vid_id] = title
            # print(f"  + Found relevant video: {title} (id={vid_id})")


        # tiny delay to be nice to the API
        time.sleep(0.3)

    # print(f"\n[INFO] Total unique relevant videos: {len(video_ids)}")
    return videos

videos = collect_videos()

# TODO: improve duplicate/blacklisted/whitelisted file checking in folder (make json file to store valid/invalid files)
# TODO: loop through entirety of each file and keep those that have a minimum stroke count using mediapipe algo

all_videos = os.listdir(path)
for i in list(videos.keys()):
    url = f"https://www.youtube.com/watch?v={i}"
    if i not in all_videos:
        print(f"downloading {videos[i]}")
        download(url,path,i)
    else:
        print(f"{videos[i]} is already in folder")
'''