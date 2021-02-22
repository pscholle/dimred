import ctypes
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(
    ROOT_DIR, 'extract_parts.so')


def main():
    testlib = ctypes.CDLL(SO_PATH)

    buff = ctypes.create_string_buffer(20)
    buff.value = b'yay'
    # p = ctypes.c_char_p(buff)
    testlib.assign_value(buff)
    print(buff.value.decode("utf-8"))


if __name__ == '__main__':
    main()
