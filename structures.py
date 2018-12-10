import pdb

class time_signature(object):
    def __init__(self, time, relation='NA', node_type='start'):
        time = extract(time)
        self.time = time
        self.time_str = '-'.join(map(str,time))
        #       in each time point, the relation is a tuple indicates its before and after relationships
        self.relation = relation
        self.type = node_type

    def __str__(self):
        return (self.time, self.relation, self.type)

    #   so what does cmp do?
    def __cmp__(self, x):
        return cmp(self.time[0], x.time[0])

    # equal defined in year level.
    def __eq__(self, other):
        return self.time[0] == other.time[0]

    #   less than
    def __lt__(self, x):
        # use year as turning point
        if self.time[0] == x.time[0] and self.time[1] == self.time[1]:
            if x.type == "mention":
                a = {'start': 0, 'mention':2, 'end': 1}
            else:
                a = {'start': 0, 'end': 1}
                #           start and end point in same time
                if self.relation != x.relation:
                    a = {'end': 0, 'start': 1}
            return a[self.type] < a[x.type]
        return self.time < x.time


def extract(time):
    if time is None:
        pdb.set_trace()
        return (None, None, None)
    t = time.split('-')
    try:
        tp = tuple(int(time) for time in t[-3:])
    except:
        pdb.set_trace()
        tp = None
    # another choice is to use year to partition the time spot.
    # v2 = True
    # if v2:
    #     tp = (int(t[-3]), 1, 1)
    return tp


class Mention(object):
    def __init__(self, sent, org_sent="", en_pair_str='', tag='NA', tag_name=None, time=None, pos1=0, pos2=0, rank=0):
        self.sent = sent
        self.org_sent = org_sent
        self.en_pair_str = en_pair_str
        self.time = time
        self.pos = (pos1, pos2)
        self.tag = tag
        self.tag_name = tag_name
        self.rank = rank

    def __lt__(self, x):
        return self.time < x.time



class Stack(object):
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    # return top element
    def peek(self):
        return self.items[len(self.items) - 1]


    def size(self):
        return len(self.items)

    def __getitem__(self, ix):
        return self.items[ix]

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

