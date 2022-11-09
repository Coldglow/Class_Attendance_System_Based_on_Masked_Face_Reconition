import os

from globals import connect_to_sql

mixedImagesDir = "data/images_mixed"


def main():
    classes = os.listdir(mixedImagesDir)
    all_local_classes_set = set()
    for item in classes:
        id = item.split("_")[0]
        name = item.split("_")[1]
        single_class_set = (id, name)
        all_local_classes_set.add(single_class_set)
    print(all_local_classes_set)

    # return all_local_classes_set

# def main():
#     classes = os.listdir(mixedImagesDir)
#     for item in classes:
#         id = item.split("_")[0]
#         name = item.split("_")[1]
#         single_class_set = (id, name)


if __name__ == '__main__':
    main()