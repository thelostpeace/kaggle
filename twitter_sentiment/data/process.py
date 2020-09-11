with open('data.csv') as f:
    for line in f:
        seg = line.strip().split(sep=',')
        if len(seg) == 4:
            print("%s\t%s\t%s\t%s" % (seg[0], seg[1], seg[2], seg[3]))
        else:
            print("%s\t%s\t%s\t%s" % (seg[0], seg[1], seg[2], ','.join(seg[3:])))
