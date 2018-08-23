
# coding: utf-8

# In[15]:


import json
import os
import wave
import pprint as pp


# In[28]:


def handler(data_dir, out_file):
    sentences = []
    durations = []
    keys = []
    
    for group in os.listdir(data_dir):
        print("group:", group)
        if group == '.DS_Store': continue
            
        group_path = os.path.join(data_dir, group)
        for speaker in os.listdir(group_path):
            print("\t speaker:", speaker)
            if group == '.DS_Store': continue
                
            labels_file = os.path.join(group_path,
                                       speaker,
                                       '{}-{}.trans.txt'.format(group, speaker))
            
            for line in open(labels_file):
                split = line.strip().split()
                file_id = split[0]
                sentence = ' '.join(split[1:]).lower()
                audio_file = os.path.join(group_path,
                                         speaker,
                                         file_id)
                audio_file += '.wav'
                # audio_file += '.flac'
                audio = wave.open(audio_file)
                duration = float(audio.getnframes())/audio.getframerate()
                audio.close()
                
                keys.append(audio_file)
                durations.append(duration)
                sentences.append(sentence)
                
        # pp.pprint(keys)
        # pp.pprint(durations)
        # pp.pprint(sentences)
        with open(output_file, 'w') as out_file:
            for i in range(len(sentences)):
                line = json.dumps({
                    'path':keys[i],
                    'time':durations[i],
                    'text':sentences[i],
                })
                
                out_file.write(line + '\n')
            
        

if __name__ == '__main__':
    path = "/Users/chenzomi/Documents/DeepLearnInAction/code/data/LibriSpeech"
    data_directory = "test-clean/"
    output_file = "test_corpus.json"
    
    data_directory = os.path.join(path, data_directory)
    output_file = os.path.join(path, output_file)
    
    handler(data_directory, output_file)

