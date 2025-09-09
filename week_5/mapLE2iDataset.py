import os
import csv
import pandas as pd
import re

# Every folder in the parent folder should only contain two other folders named Videos and Annotation_files
# Install all libraries before using the function

def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def mapLE2iDataset(parentFolder):
    """
    Traverses dataset folders and maps videos to their annotations.

    Args:
        parentFolder (str): path to the root folder containing all environment folders

    Returns:
        infoList (list of lists): rows with
            [video_path, fall_label, startFrame, endFrame]
    """
    infoList = []

    # List environment folders
    envFolders = [d for d in os.listdir(parentFolder)
                  if os.path.isdir(os.path.join(parentFolder, d))]

    for env in envFolders:
        envPath = os.path.join(parentFolder, env)
        videoFolder = os.path.join(envPath, "Videos")
        annotFolder = os.path.join(envPath, "Annotation_files")

        # Skip if subfolders missing
        if not (os.path.isdir(videoFolder) and os.path.isdir(annotFolder)):
            print(f"Skipping {envPath}, missing subfolders.")
            continue

        # Get annotation files and sort naturally
        annotFiles = sorted([f for f in os.listdir(annotFolder) if f.endswith(".txt")],
                            key=natural_sort_key)
        arrLabel = []
        fallFrames = []

        for annot in annotFiles:
            annotPath = os.path.join(annotFolder, annot)
            startFrame, endFrame = readFirstNumbers(annotPath)
            fallFrames.append([startFrame, endFrame])
            arrLabel.append(1 if startFrame > 0 else 0)

        # Get video files and sort naturally
        videoFiles = sorted([f for f in os.listdir(videoFolder) if f.endswith(".avi")],
                            key=natural_sort_key)

        # Map videos to labels
        for k, vFile in enumerate(videoFiles):
            vPath = os.path.join(videoFolder, vFile)
            infoList.append([vPath, arrLabel[k], fallFrames[k][0], fallFrames[k][1]])

    return infoList


def readFirstNumbers(filePath):
    """Read first two numbers from annotation file, like MATLAB's fgetl + str2double."""
    startFrame, endFrame = 0, 0
    with open(filePath, "r") as f:
        lines = f.readlines()
        if len(lines) > 0:
            try:
                startFrame = float(lines[0].strip())
            except ValueError:
                startFrame = 0
        if len(lines) > 1:
            try:
                endFrame = float(lines[1].strip())
            except ValueError:
                endFrame = 0
    return int(startFrame), int(endFrame)


def exportToCSV(infoList, filename="video_info.csv"):
    headers = ["video_path", "fall_label", "startFrame", "endFrame"]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(infoList)
    print(f"CSV saved as {filename}")


def exportToExcel(infoList, filename="video_info.xlsx"):
    headers = ["video_path", "fall_label", "startFrame", "endFrame"]
    df = pd.DataFrame(infoList, columns=headers)
    df.to_excel(filename, index=False)
    print(f"Excel saved as {filename}")


def printTable(infoList):
    """Print a neat aligned table in console (like MATLAB version)."""
    headers = ["video_path", "fall_label", "startFrame", "endFrame"]
    maxPathLength = max(len(row[0]) for row in infoList) if infoList else len(headers[0])

    print(f"{headers[0]:<{maxPathLength}} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<10}")
    print("-" * (maxPathLength + 36))

    for row in infoList:
        print(f"{row[0]:<{maxPathLength}} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10}")


if __name__ == "__main__":
    parentFolder = input("Enter the path to the parent folder: ")
    infoList = mapLE2iDataset(parentFolder)

    # Export to CSV and Excel
    exportToCSV(infoList, "video_info.csv")
    exportToExcel(infoList, "video_info.xlsx")

    # Print table in console
    printTable(infoList)