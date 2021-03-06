import os
import numpy as np
from moviepy.video.io.ffmpeg_reader import *


def _list_delete(a, b):
    return list(filter(lambda i: i not in b, a))


class UCFVideo(FFMPEG_VideoReader):

    '''
    def __init__(self, filename, print_infos=False, bufsize=None,
                 pix_fmt="rgb24", check_duration=True):

        self.filename = filename
        infos = ffmpeg_parse_infos(filename, print_infos, check_duration)
        self.fps = infos['video_fps']
        self.size = infos['video_size']
        self.duration = infos['video_duration']
        self.ffmpeg_duration = infos['duration']
        self.nframes = infos['video_nframes']

        self.infos = infos

        self.pix_fmt = pix_fmt
        if pix_fmt == 'rgba':
            self.depth = 4
        else:
            self.depth = 3

        if bufsize is None:
            w, h = self.size
            bufsize = self.depth * w * h + 100

        self.bufsize = bufsize
        self.pos = 1
        self.width = self.size[0]
        self.height = self.size[1]
    '''

    def get_length(self, frames=0, secondes=0):
        if frames != 0:
            return int(self.nframes / frames)
        elif secondes != 0:
            dis = int(self.fps * secondes + 0.00001)
            return int(self.nframes / dis)
        else:
            return self.nframes

    def read_frames(self, seq_length=0, frames=0, secondes=0):
        if seq_length == 0:
            seq_length = self.nframes

        video_length = self.get_length(frames, secondes)
        if seq_length > video_length:
            seq_length = video_length

        gap = 0
        if frames != 0:
            gap = frames - 1
        elif secondes != 0:
            gap = int(self.fps * secondes + 0.00001) - 1
            if gap < 0: gap = 0

        w, h = self.size
        data = np.zeros((seq_length, h, w, self.depth), dtype=np.uint8)

        extend = seq_length + gap * (seq_length - 1)
        space = self.nframes - extend
        start = np.random.randint(space + 1)

        self.initialize()
        self.pos = 1
        self.skip_frames(start)
        data[0, :] = self.read_frame()
        self.pos += 1
        for i in range(1, seq_length):
            self.skip_frames(gap)
            data[i, :] = self.read_frame()
            self.pos += 1
        return data, seq_length


class VideoInput:

    VIDEO_HEIGHT = 240
    VIDEO_WIDTH = 320

    def __init__(self, path):
        self.root = path
        self.classes = os.listdir(path)
        self.files = []
        self.sep = [0]
        self.group = None
        self.selected_classes = None

        for c in self.classes:
            sub_path = os.path.join(path, c)
            files = os.listdir(sub_path)
            self.files += files
            self.sep.append(self.sep[-1] + len(files))

    def grouping(self, train_rate, validation_rate, test_rate):
        self.group = {'train': [], 'validation': [], 'test':[]}
        classes = self.selected_classes or range(len(self.classes))
        for i in classes:
            files = list(range(self.sep[i], self.sep[i+1]))
            train_size = int(len(files) * train_rate)
            validation_size = int(len(files) * validation_rate)
            train = list(np.random.choice(files, train_size, False))
            temp = _list_delete(files, train)
            validation = list(np.random.choice(temp, validation_size, False))
            test = _list_delete(temp, validation)
            if test_rate == 0 and len(test) != 0:
                for item in test:
                    train.append(item) if np.random.randint(2) else validation.append(item)
                test = []

            self.group['train'] += train
            self.group['validation'] += validation
            self.group['test'] += test

        return (len(self.group['train']), len(self.group['validation']), len(self.group['test']))

    def select_sub_collection(self, n_classes, log_file=None):
        choices = np.random.choice(range(len(self.classes)), n_classes, False)
        self.selected_classes = sorted(choices)

    def get_data(self, group, batch, seq_length=0,
                 random_mode=False, frames=0, secondes=0,
                 size=(VIDEO_HEIGHT, VIDEO_WIDTH)):
        '''Get a batch of video data.

        Parameters:
        ---------------------
        group: str
               choice from "train", "validation" or "test".
        batch: int
               batch size.
        seq_length: int
               length of video frames.
               if not set (None), return all the frames in a video file.
        ...

        Return:
        ---------------------
        data: 5D ndarray
              in shape "batch x max_seq_length x height x weight x channel"
        videos_length: 1D ndarray whith size = batch
              the length of video frames of each sample.
        label: 1D ndarray whith size = batch
              the class label of each sample.
        '''

        assert size[0] <= self.VIDEO_HEIGHT
        assert size[1] <= self.VIDEO_WIDTH

        if batch == "all":
            selected_files = np.array(self.group[group], dtype=np.int32)
            batch = len(self.group[group])
        else:
            selected_files = np.random.choice(self.group[group], batch, False)

        if random_mode:
            frames = [0] * batch
            secondes = np.random.randint(1, 11, batch) / 10
        else:
            frames = [frames] * batch
            secondes = [secondes] * batch

        label = np.zeros(batch, dtype=np.uint8)
        for sep in self.sep[1:-1]:
            label[selected_files >= sep] += 1

        videos_length = np.zeros(batch, dtype=np.uint32)
        for i in range(batch):
            video_file = selected_files[i]
            video_class = label[i]
            file_path = os.path.join(self.root, self.classes[video_class], self.files[video_file])

            ucf = UCFVideo(file_path)
            videos_length[i] = ucf.get_length(frames[i], secondes[i])
            del ucf

        max_video_length = videos_length.max()
        if seq_length == 0 :
            seq_length = max_video_length

        h, w = size
        data = np.zeros(shape=(seq_length, batch, h, w, 3), dtype=np.uint8)
        for i in range(batch):
            video_file = selected_files[i]
            video_class = label[i]
            file_path = os.path.join(self.root, self.classes[video_class], self.files[video_file])

            ucf = UCFVideo(file_path)
            video_data, length = ucf.read_frames(seq_length, frames[i], secondes[i])
            width, height = ucf.size
            h_start = int((height - h) / 2)
            w_start = int((width - w) / 2)
            data[0:length, i, :] = video_data[:, h_start:(h_start + h), w_start:(w_start + w)]
            videos_length[i] = length
            del ucf

        if self.selected_classes is not None:
            label = np.array(list(map(lambda l: np.where(self.selected_classes == l)[0][0] , label)))

        return (data, videos_length, label)

    def save(self, file_path):
        if self.group == None:
            return
        with open(file_path, 'w') as log:
            classes = self.selected_classes or range(len(self.classes))
            log.write("%i" % (len(classes)) + os.linesep)
            for c in classes:
                log.write("%i %s" % (c, self.classes[c]) + os.linesep)
            log.write("train:" + os.linesep)
            log.write(" ".join(list(map(lambda i: str(i), self.group["train"]))) + os.linesep)
            log.write("validation:" + os.linesep)
            log.write(" ".join(list(map(lambda i: str(i), self.group["validation"]))) + os.linesep)
            log.write("test:" + os.linesep)
            log.write(" ".join(list(map(lambda i: str(i), self.group["test"]))) + os.linesep)

    def load(self, file_path):
        with open(file_path, 'r') as log:
            n_classes = int(log.readline())
            if n_classes == 101:
                self.selected_classes = None
            else:
                self.selected_classes = []
            for i in range(n_classes):
                line = log.readline()
                if n_classes != 101:
                    self.selected_classes.append(int(line.split(' ')[0]))
            self.group = {}
            log.readline()
            self.group["train"] = list(map(lambda i: int(i), log.readline().split(' ')))
            log.readline()
            self.group["validation"] = list(map(lambda i: int(i), log.readline().split(' ')))
            log.readline()
            self.group["test"] = list(map(lambda i: int(i), log.readline().split(' ')))

'''
if __name__ == "__main__":

    video_input = VideoInput("/home/filick/workspace/VideoClassification/UCF-101")
    result = video_input.grouping(0.7, 0.1, 0.2)
    
    print(result, sum(result), video_input.sep[-1])
    total = video_input.group['train'] + video_input.group['validation'] + video_input.group['test']
    total = np.array(sorted(total))
    check = total - np.arange(0, video_input.sep[-1])
    print((check != 0).sum())

    data, seq_len, label = video_input.get_data("train", 100, random_mode=True, seq_length=30)
    print(seq_len, label)
    import matplotlib.pyplot as plt
    plt.imshow(data[0, 0, :])
    plt.show()
'''