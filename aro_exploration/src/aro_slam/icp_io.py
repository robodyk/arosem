import json
import numpy as np
import os
import pickle


def makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


def read_binary(path):
    assert isinstance(path, str)
    with open(path, 'rb') as f:
        return f.read()


def write_binary(path, s):
    assert isinstance(path, str)
    assert isinstance(s, (bytes, str))
    makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        f.write(s)


def read_text(path, encoding='utf-8'):
    assert isinstance(path, str)
    assert isinstance(encoding, str)
    return read_binary(path).decode(encoding)


def write_text(path, s, encoding='utf-8'):
    assert isinstance(path, str)
    if not isinstance(s, str):
        print(type(s))
    assert isinstance(s, str)
    assert isinstance(encoding, str)
    makedirs(os.path.dirname(path))
    write_binary(path, s.encode(encoding))


def read_object(path):
    assert isinstance(path, str)
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_object(path, obj):
    assert isinstance(path, str)
    makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_json(path):
    assert isinstance(path, str)
    with open(path, 'r') as fp:
        return json.load(fp)


def write_json(path, obj):
    assert isinstance(path, str)
    makedirs(os.path.dirname(path))
    with open(path, 'w') as fp:
        json.dump(obj, fp, indent=1, sort_keys=True)


def read_poses(path):
    poses = np.genfromtxt(path, delimiter=', ', skip_header=True)
    ids = np.genfromtxt(path, delimiter=', ', dtype=str, skip_header=True)[:, 0].tolist()
    # assert ids == list(range(len(ids)))
    poses = poses[:, 2:]
    poses = poses.reshape((-1, 4, 4))
    poses = dict(zip(ids, poses))
    return poses


def read_cloud(npz_file):
    cloud = np.load(npz_file)['cloud']
    if cloud.ndim == 2:
        cloud = cloud.reshape((-1,))
    return cloud
