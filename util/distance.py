import difflib

def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        # print(tag, i1, i2, j1, j2)
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost


if __name__ == '__main__':
    print(GetEditDistance('aa', 'cab'))
