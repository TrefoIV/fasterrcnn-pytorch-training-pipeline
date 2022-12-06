from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)


train_dataset = create_train_dataset("./data/scalograms/train", "./data/scalograms/train", 10000, 2249, ["__background__", "sequences"], discard_negative=False)

print(f"Immagini nel train {len(train_dataset)}")
for i in range(len(train_dataset)):
    _ = train_dataset.load_image_and_labels(i)
    if i % 10 == 0:
        print(i)


'''
with open("file_rimossi", "r") as f:
    xmls_empty = set()
    xmls_not_found = set()
    removed = 0
    line = f.readline().replace("\n", "")
    while line.endswith(".jpg"):
        xmls_empty.add(line.split()[1])
        line = f.readline().replace("\n", "")
    print(f"total xmls_empty removed {len(xmls_empty)}")
    while line != "":
        if not line.endswith("image"):
            xml = line.split()[0]
            if xml not in xmls_empty:
                xmls_not_found.add(xml)
            else:
                print(f"Removing {xml} from set")
                xmls_empty.remove(xml)
                removed += 1
        line = f.readline().replace("\n", "")

    print("-"*50)
    print(f"Remaining xmls_empty {len(xmls_empty)}")
    print(f"Total removed {removed}")
    for xml in xmls_empty:
        print(xml)

    print("-"*50)
    print(f"xmls_not_founded {len(xmls_not_found)}")
    for xml in xmls_not_found:
        print(xml)
import sys
sys.exit(0)
'''