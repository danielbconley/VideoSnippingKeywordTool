from faster_whisper import WhisperModel
import subprocess
import imageio_ffmpeg
import os
import sys

#------------CONFIGURATION-------------#
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
EXCLUDED_CHARACTERS = ",.!?'" # characters excluded from stripped strings
CHATTER_BUFFER = 10 # determines the amount of words used as a buffer if multiple people are talking at once
MEDIA_EXTENSIONS = (".mp4", ".mov", ".m4a", ".mp3", ".mkv") # file types shown in the file selection menu
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
    nonKeywords = 0
    for segment in segments:
        for word in segment.words:
            wordObjectStringStripped = word.word.lstrip().lower().strip(EXCLUDED_CHARACTERS)
            print(f'Debugging: The transcribed string is {wordObjectStringStripped}')
            # logic to store arrays full of slice points for processing
            if wordObjectStringStripped == keywordStripped: # if the current word is equal to the keyword, writeCondition is True to prime appending timestamps
                writeCondition = True
                nonKeywords = 0
                print("Debugging: Keyword identified")
            elif writeCondition == True and wordObjectStringStripped == "start": # case for appending start of timestamp
                print("Debugging: Appending start timestamp")
                start_points.append(word.start)
                writeCondition = False
                nonKeywords = 0
            elif writeCondition == True and wordObjectStringStripped == "end": # case for appending end of timestamp
                print("Debugging: Appending end timestamp")
                end_points.append(word.end)
                writeCondition = False
                nonKeywords = 0
            elif writeCondition == True: # case to handle user unintentionally saying the keyword
                nonKeywords += 1
                if nonKeywords >= CHATTER_BUFFER: # if the amount of words after the keyword exceed or are equal to the buffer, the writeCondition is set back to False
                    print("Debugging: Buffer exceeded, deactivating writeCondition")
                    writeCondition = False
                    nonKeywords = 0

    print(start_points, "\n", end_points)
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

if __name__ == "__main__":
    main()
