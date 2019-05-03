import json

pro_dir = '/home/lijin/PycharmProjects/CRNN_CTC/'
label_dir = pro_dir + '/json/label.json'

f = open(label_dir, 'r')

label_dict = json.load(f)
if label_dict is None:
    print('kongde')
for s in label_dict.keys():
    label_dict[s] = label_dict[s].replace(' ', '')
g = open(pro_dir+'new_label.json','w')

json.dump(label_dict,g)
f.close()
g.close()