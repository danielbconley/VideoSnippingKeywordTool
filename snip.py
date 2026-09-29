from faster_whisper import WhisperModel
import subprocess
import imageio_ffmpeg
import os
import sys
import time

#  TO BE IMPLEMENTED:
# -swap debugging print statements for logging - STATUS: UNIMPLEMENTED
# -handle case where "start" is said twice. when clip start is said a second time, its ignored - STATUS: IMPLEMENTED, BEHAVIOR NOT VERIFIED.
# -handle case where string has - in front of word STATUS: UNIMPLEMENTED

#------------CONFIGURATION-------------#
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
EXCLUDED_CHARACTERS = ",.!?'" # characters excluded from stripped strings
CHATTER_BUFFER = 5 # determines the amount of words used as a buffer if multiple people are talking at once
CLIP_START_BUFFER = 1 # used as a buffer to avoid including the keyword in video playback
CLIP_END_BUFFER = 2 # used as a buffer to avoid including the keyword in video playback
MEDIA_EXTENSIONS = (".mp4", ".mov", ".m4a", ".mp3", ".mkv") # file types shown in the file selection menu
START_PHONETIC_SIMILARITIES = {"sart", "shart", "stark", "star"} # set to handle transcription innacuracies for start
END_PHONETIC_SIMILARITIES = {"in", "n", "en", "and"} # set to handle transcription innacuracies for end
model_size = "small" # select model size - note: larger models are slower but more accurate
model = WhisperModel(model_size, device="cpu", compute_type="int8") # select whether to use cpu or gpu for processing
#--------------------------------------#

def main():
    keyword = keywordSelect()
    input_file = fileSelect()

    segments = transcribe(input_file)
    start_points, end_points = findSlicePoints(segments, keyword)
    sliceVideo(input_file, start_points, end_points)

def keywordSelect():
    selection = 2
    while selection == 2:
            keyword = input("Enter your keyword: ")
            selection = int(input(f"Your keyword is: {keyword}. Press 1 to proceed, or 2 to re-enter keyword: "))
    return keyword

def fileSelect():
    # a path passed on the command line (python snip.py "path\to\video.mp4") skips the menu
    if len(sys.argv) > 1:
        return sys.argv[1]

    # otherwise, list the media files in the same folder as this script and pick one by number
    folder = os.path.dirname(os.path.abspath(__file__))
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(MEDIA_EXTENSIONS))
    if not files:
        sys.exit(f"No media files found in {folder}")

    for i, f in enumerate(files, start=1):
        print(f"{i}. {f}")

    while True:
        try:
            choice = int(input("Select a file by number: "))
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= choice <= len(files):
            return os.path.join(folder, files[choice - 1])
        print(f"Please enter a number between 1 and {len(files)}.")

def transcribe(input_file):
    segments, info = model.transcribe(input_file, word_timestamps=True)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    return segments

def findSlicePoints(segments, keyword):
    start_points = []
    end_points = []
    keywordStripped = keyword.lower().strip(EXCLUDED_CHARACTERS)
    print(f'Debugging: The stripped keyword string is {keywordStripped}')
    writeCondition = False
    clip_ongoing = False
    nonKeywords = 0
    start_time = time.perf_counter()

    for segment in segments:
        for word in segment.words:
            wordObjectStringStripped = word.word.lstrip().lower().strip(EXCLUDED_CHARACTERS)
            print(f'Debugging: The transcribed string is {wordObjectStringStripped}')
            # logic to store arrays full of slice points for processing
            if wordObjectStringStripped == keywordStripped: # if the current word is equal to the keyword, writeCondition is True to prime appending timestamps
                writeCondition = True
                nonKeywords = 0
                print(f"Debugging: Keyword identified at {word.start:.2f}s")

            elif clip_ongoing == True and writeCondition == True and (wordObjectStringStripped == "start" or wordObjectStringStripped in START_PHONETIC_SIMILARITIES):
                # case to handle "start" being said twice
                writeCondition = False

            elif clip_ongoing == False and writeCondition == True and (wordObjectStringStripped == "end" or wordObjectStringStripped in END_PHONETIC_SIMILARITIES):
                # case to handle "end" being said twice
                writeCondition = False

            elif clip_ongoing == False and writeCondition == True and (wordObjectStringStripped == "start" or wordObjectStringStripped in START_PHONETIC_SIMILARITIES): 
                # case for appending start of timestamp
                print(f"Debugging: Appending start timestamp {word.start:.2f}.s")
                if wordObjectStringStripped in START_PHONETIC_SIMILARITIES:
                    print(f"Debugging: Phonetic similarity {wordObjectStringStripped} identified")
                start_points.append(word.start + CLIP_START_BUFFER)
                writeCondition = False
                nonKeywords = 0
                clip_ongoing = True

            elif clip_ongoing == True and writeCondition == True and (wordObjectStringStripped == "end" or wordObjectStringStripped in END_PHONETIC_SIMILARITIES): 
                # case for appending end of timestamp
                print(f"Debugging: Appending end timestamp {word.start}s")
                if wordObjectStringStripped in END_PHONETIC_SIMILARITIES:
                    print(f"Debugging: Phonetic similarity {wordObjectStringStripped} identified")
                end_points.append(word.start - CLIP_END_BUFFER)
                writeCondition = False
                nonKeywords = 0
                clip_ongoing = False

            elif writeCondition == True: # case to handle user unintentionally saying the keyword
                nonKeywords += 1
                if nonKeywords >= CHATTER_BUFFER: # if the amount of words after the keyword exceed or are equal to the buffer, the writeCondition is set back to False
                    print("Debugging: Buffer exceeded, deactivating writeCondition")
                    writeCondition = False
                    nonKeywords = 0

    time_elapsed = time.perf_counter() - start_time
    print(f"Debugging: Speech recognition took {time_elapsed} seconds")
    print("Debugging: Timestamps are:")
    for s, e in zip(start_points, end_points):
        print(f"{s} - {e}")
    return start_points, end_points

def sliceVideo(input_file, start_points, end_points):
    # slices at timestamps and exports each clip next to the input file
    name, ext = os.path.splitext(input_file)
    for i, (s, e) in enumerate(zip(start_points, end_points)):
        if e <= s:
            print(f"Debugging: Skipping clip {i}, end {e} is not after start {s}")
            continue
        output_file = f"{name}_sliced_{i}{ext}"
        print(f"Debugging: Exporting {output_file}")
        cut_clip(input_file, s, e, output_file, exact=False)

def cut_clip(input_file, start, end, output_file, exact=False):
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", input_file,
        "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a:0",
    ]
    if exact:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-avoid_negative_ts", "make_zero", output_file]
    subprocess.run(cmd, check=True)

def concatenate_clips():
    pass

if __name__ == "__main__":
    main()
