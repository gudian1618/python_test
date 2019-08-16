def power_seq(func, seq):
    return [func(i) for i in seq]
    pass


def pingfang(x):
    return x ** 2


if __name__ == '__main__':
    num_seq = [11, 3.14, 2.98]
    r = power_seq(pingfang, num_seq)
    print(num_seq)
    print(r)
