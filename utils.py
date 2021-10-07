def get_congruency(loc1, loc2):
    (x1, y1), (x2, y2) = loc1, loc2
    if ((x1==x2) or (y1==y2)):
        cong = 0
    else:
        cong = 1 if (x1<x2) == (y1<y2) else -1
    return cong

def log(tr_acc, te_acc, an_acc):
    msg = "Train: {}, Test: {}, Analysis: {}".format(tr_acc, te_acc, an_acc)
    print(msg)
    return msg