from faster_whisper import WhisperModel

#------------CONFIGURATION-------------#
KEYWORD = "clip" # select keyword for slice points
EXCLUDED_CHARACTERS = ",.!?'" # characters excluded from stripped strings
CHATTER_BUFFER = 10 # determines the amount of words used as a buffer if multiple people are talking at once **IMPLEMENT LATER**
model_size = "medium" # select model size - note: larger models are faster but have more hardware needs
model = WhisperModel(model_size, device="cpu", compute_type="int8") # select whether to use cpu or gpu for processing


segments, info = model.transcribe("Recording.m4a", word_timestamps=True)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

slice_points = []
keywordStripped = KEYWORD.lower().strip(EXCLUDED_CHARACTERS)
print(f'Debugging: The stripped keyword string is {keywordStripped}')
writeCondition = False
for segment in segments:
    for word in segment.words:
        print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))

        
        wordObjectStringStripped = word.word.lstrip().lower().strip(EXCLUDED_CHARACTERS)

        print(f'Debugging: The stripped transcription string is {wordObjectStringStripped}')

        # logic to store an array full of slice points for post-processing
        if wordObjectStringStripped == keywordStripped:
            writeCondition = True
            print("Debugging: Keyword identified")
        elif writeCondition == True and wordObjectStringStripped == "start":
            print("Debugging: Appending start timestamp")
            slice_points.append(word.start)
            writeCondition = False
        elif writeCondition == True and wordObjectStringStripped == "end":
            print("Debugging: Appending end timestamp")
            slice_points.append(word.end)
            writeCondition = False

print(slice_points)

        

        


