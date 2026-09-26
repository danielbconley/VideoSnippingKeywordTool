from faster_whisper import WhisperModel
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

#------------CONFIGURATION-------------#
KEYWORD = "clip" # select keyword for slice points
EXCLUDED_CHARACTERS = ",.!?'" # characters excluded from stripped strings
CHATTER_BUFFER = 10 # determines the amount of words used as a buffer if multiple people are talking at once
model_size = "medium" # select model size - note: larger models are slower but more accurate
model = WhisperModel(model_size, device="cpu", compute_type="int8") # select whether to use cpu or gpu for processing
#--------------------------------------#
def main(INPUT_FILE): 
    segments, info = model.transcribe(INPUT_FILE, word_timestamps=True)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    start_points = []
    end_points = []
    keywordStripped = KEYWORD.lower().strip(EXCLUDED_CHARACTERS)
    print(f'Debugging: The stripped keyword string is {keywordStripped}')
    writeCondition = False
    nonKeywords = 0
    for segment in segments:
        for word in segment.words:
            wordObjectStringStripped = word.word.lstrip().lower().strip(EXCLUDED_CHARACTERS)
            print(f'Debugging: The stripped transcribed string is {wordObjectStringStripped}')
            # logic to store arrays full of slice points for processing
            if wordObjectStringStripped == keywordStripped: # if the current word is equal to the keyword, writeCondition is True to prime appending timestamps
                writeCondition = True
                print("Debugging: Keyword identified")
            elif writeCondition == True and wordObjectStringStripped == "start": # case for appending start of timestamp
                print("Debugging: Appending start timestamp")
                start_points.append(word.start)
                writeCondition = False
            elif writeCondition == True and wordObjectStringStripped == "end": # case for appending end of timestamp
                print("Debugging: Appending end timestamp")
                end_points.append(word.end)
                writeCondition = False
            elif writeCondition == True: # case to handle user unintentionally saying the keyword
                nonKeywords += 1
                if nonKeywords >= CHATTER_BUFFER: # if the amount of words after the keyword exceed or are equal to the buffer, the writeCondition is set back to False
                    print("Debugging: Buffer exceeded, deactivating writeCondition")
                    writeCondition = False
                    nonKeywords = 0
            
    print(start_points, "\n", end_points)

    # slices at timestamps and exports the video 
    for s, e in zip(start_points, end_points):
        ffmpeg_extract_subclip(inputfile=INPUT_FILE, start_time=s, end_time=e, outputfile=f"sliced_{INPUT_FILE}")

if __name__ == "__main__":
    main()

    
    

        


