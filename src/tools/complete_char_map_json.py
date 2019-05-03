import json

pro_dir = '/home/lijin/PycharmProjects/CRNN_CTC/'
map_dir = pro_dir + '/json/char_map.json'
label_dir = pro_dir + '/json/label.json'
f = open(map_dir, 'r')
char_dict = json.load(f)
l = open(label_dir, 'r')
label_dict = json.load(l)
for labels in label_dict.values():
    labels = labels.lower()
    print(labels)
    for char in labels:
        if not char in char_dict.keys():
            char_dict[char] = len(char_dict)

m = open(pro_dir + 'new_char_map.json','w')
json.dump(char_dict,m)
f.close()
m.close()
l.close()