################## DON'T USE indent is json dump!!! ##############
import lyricsgenius
import json
import time
import random

from config import CLIENT_ACCESS_TOKEN
from song_list import song_pair_list, song_pair_list2

genius = lyricsgenius.Genius(CLIENT_ACCESS_TOKEN)

# song = genius.search_song(title="Bohemian Rhapsody",
#                            artist="Queen",
#                         get_full_info=True)

count = 0
success = 0
failed = 0
list_len = len(song_pair_list)

with open("output.jsonl", "a", encoding="utf-8") as f:
    for song_pair in song_pair_list2:
        song_name = song_pair[0]
        artist_name = song_pair[1]
        count += 1

        print(f"[{count}/{list_len}] {song_name} - {artist_name}")

        try:
            song = genius.search_song(title=song_name,
                                artist=artist_name,
                                get_full_info=True)

            if song is None:
                print(f"Number {count} failed")
                failed += 1
                continue

            entry = {
                    "title": song.title,
                    "artist": artist_name,
                    "lyrics": song.lyrics,
                    "lyrics_length": len(song.lyrics),
                    "metadata": song.to_dict()
                }

            f.write(json.dumps(entry, indent=4, ensure_ascii=False) + ("\n"))
            f.flush()
            success += 1
            print(f"Success {song_name}, {artist_name} ")

        except Exception  as e:
            failed += 1
            print(f"Exception for song: {song_name}, artist: {artist_name}")
            print(e)
            print(f"Song Number: {count}")
            continue


        time.sleep(random.randint(0, 3))
    

print("\n" + "=" * 30)
print("Finished")
print(f"Total:   {count}")
print(f"Success: {success}")
print(f"Failed:  {failed}")
print("=" * 30)
