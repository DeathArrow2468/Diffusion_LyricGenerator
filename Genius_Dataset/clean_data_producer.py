####### Makes a JSONL file of only artist, lyrics and description #######
import json

INPUT_FILE = "output.jsonl"
OUTPUT_FILE = "raw_data.jsonl"

count = 0
success = 0
failed = 0

decoder = json.JSONDecoder()

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    text = infile.read()
    position = 0
    length = len(text)

    while position < length:

        while position < length and text[position].isspace():
            position += 1

        if position >= length:
            break

        try:
            data, end = decoder.raw_decode(text, position)
            position = end
            count += 1

            entry = {
                "artist": data.get("artist"),
                "song_name": data.get("title"),
                "lyrics": data.get("lyrics"),
                "description": (
                    data.get("metadata", {})
                        .get("description", {})
                        .get("plain")
                )
            }

            if not entry["artist"] or not entry["song_name"] or not entry["lyrics"]:
                failed += 1
                print(f"Skipping entry {count}: missing required data")
                continue

            outfile.write(
                json.dumps(entry, ensure_ascii=False) + "\n"
            )

            success += 1

            if success % 100 == 0:
                print(
                    f"Processed: {count:,} | "
                    f"Success: {success:,} | "
                    f"Failed: {failed:,}"
                )

        except json.JSONDecodeError as e:
            failed += 1
            print(f"JSON error around position {position:,}: {e}")
            print(repr(text[position:position + 500]))
            break

print("\n" + "=" * 40)
print("Cleaning finished")
print(f"Total:   {count:,}")
print(f"Success: {success:,}")
print(f"Failed:  {failed:,}")
print("=" * 40)