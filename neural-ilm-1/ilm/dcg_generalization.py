"""
handle generalizing dcg rules, to form fewer, simpler rules
"""
import my_dcg as dcg


def merge_categories(rules, r1, r2):
    if r1.category == r2.category:
        return None
    if r1.out != r2.out:
        return None
    # print('merging ', r1.category, 'and', r2.category)

    # print('len(rules)', len(rules))
    # print(rules)
    fold_from = r2.category
    fold_to = r1.category
    rules = [r for r in rules if r != r2]
    for rule in rules:
        if rule.category == fold_from:
            rule.category = fold_to
        for item in rule.out:
            if isinstance(item, dcg.NonTerminal):
                if item.category == fold_from:
                    item.category = fold_to
    # print('len(rules)', len(rules))
    # print(rules)
    return rules


def merge_chunked_bothrules(rules, r1, r2):
    if r1.category != r2.category:
        return None
    cat = r1.category
    assert cat == r2.category
    if r1.arg.__class__ != dcg.ab or r2.arg.__class__ != dcg.ab:
        return None
    num_diffs = 0
    if r1.arg.ai is None or r1.arg.bi is None or r2.arg.ai is None or r2.arg.bi is None:
        return None
    # print('r1', r1)
    # print('r2', r2)
    if r1.arg.ai != r2.arg.ai:
        # assert r1.arg.bi == r2.arg.bi
        if r1.arg.ai is None or r2.arg.ai is None:
            return None
        num_diffs += 1
        diffarg1 = dcg.a(r1.arg.ai)
        diffarg2 = dcg.a(r2.arg.ai)
        vararg = dcg.a(None)
        inarg = dcg.ab(None, r1.arg.bi)
    if r1.arg.bi != r2.arg.bi:
        if r1.arg.bi is None or r2.arg.bi is None:
            return None
        # assert r1.arg.ai == r2.arg.ai
        num_diffs += 1
        diffarg1 = dcg.b(r1.arg.bi)
        diffarg2 = dcg.b(r2.arg.bi)
        vararg = dcg.b(None)
        inarg = dcg.ab(r1.arg.ai, None)
    if num_diffs != 1:
        return None
    oldres_1 = dcg.translate(rules, dcg.NonTerminal(category=cat, arg=r1.arg))
    oldres_2 = dcg.translate(rules, dcg.NonTerminal(category=cat, arg=r2.arg))

    # first check that the non terminals match, and that the order of strings/nonterminals matches
    if len(r1.out) != len(r2.out):
        return None
    num_diff_strings = 0
    diff_pos = None
    for i in range(len(r1.out)):
        item1 = r1.out[i]
        item2 = r2.out[i]
        if item1.__class__ != item2.__class__:
            return None
        if isinstance(item1, str):
            if item1 != item2:
                num_diff_strings += 1
                diff_pos = i
    if num_diff_strings != 1:
        return None
    # common_left = []
    # common_right = []
    # if diff_pos > 0:
    common_left = r1.out[:diff_pos]
    common_right = r1.out[diff_pos + 1:]
    # print('common_left', common_left)
    # print('common_right', common_right)
    s1 = r1.out[diff_pos]
    s2 = r2.out[diff_pos]
    # print('diff', s1, s2)

    # go through, find first difference, from each end
    first_diff = 0
    last_diff = 1
    for i in range(max(len(s1), len(s2))):
        if len(s1) - 1 < i or len(s2) - 1 < i:
            first_diff = i
            break
        if s1[i] != s2[i]:
            first_diff = i
            break

    for i in range(1, max(len(s1), len(s2))):
        # print('i', i, s1[-i], s2[-i])
        if len(s1) < i or len(s2) < i:
            # print('length break', i)
            last_diff = i
            break
        if s1[-i] != s2[-i]:
            last_diff = i
            break

    if first_diff == 0 and last_diff == 1:
        return None

    shortest_s_len = min(len(s1), len(s2))
    if first_diff + last_diff - 1 > shortest_s_len:
        # ignore one of them, because we're overlapping...
        if last_diff - 1 > first_diff:
            first_diff = 0
        else:
            last_diff = 1

    s_start = ''
    if first_diff > 0:
        s_start = s1[:first_diff]
        assert s_start == s2[:first_diff]
    s_end = ''
    # print('s1', s1)
    # print('s2', s2)
    if last_diff > 1:
        s_end = s1[1 - last_diff:]
        # print('last_diff', last_diff, 's_end', s_end)
        assert s_end == s2[1 - last_diff:]
    # print(s_start, s_end)
    if last_diff > 1:
        s1_mid = s1[first_diff:1 - last_diff]
        s2_mid = s2[first_diff:1 - last_diff]
    else:
        s1_mid = s1[first_diff:]
        s2_mid = s2[first_diff:]

    # print(s_start, s_end)
    # print('s1_mid', s1_mid, 's2_mid', s2_mid)
    # asdf

    # print('old rules', rules)
    rules = [r for r in rules if r != r1 and r != r2]

    cat = r1.category
    new_cat = 'C' + str(len(rules))
    # cat1 = 'C' + str(len(rules)) + '_1'
    # cat2 = 'C' + str(len(rules)) + '_2'
    # cat_common = 'C' + str(len(rules)) + '_c'
    # print('new cat', new_cat)
    # if s1_mid != '':
    new_rule1 = dcg.Rule(
        category=new_cat, arg=diffarg1, out=[s1_mid])
    # print(new_rule1)
    rules.append(new_rule1)
    # if s2_mid != '':
    new_rule2 = dcg.Rule(
        category=new_cat, arg=diffarg2, out=[s2_mid])
    # print(new_rule2)
    rules.append(new_rule2)
    # print('len(rules)', len(rules))
    # print('len(rules)', len(rules))
    # common_left = []
    # common_right = []
    # if diff_pos > 0:
    #     common_left = 
    if s_start != '':
        common_left.append(s_start)
    if s_end != '':
        common_right.append(s_end)
    new_rule_common = dcg.Rule(
        category=cat,
        out=common_left + 
            [dcg.NonTerminal(category=new_cat, arg=vararg)] +
            common_right,
        arg=inarg
    )
    # print('new_rule_common', new_rule_common)
    rules.append(new_rule_common)

    newres_1 = dcg.translate(rules, dcg.NonTerminal(category=cat, arg=r1.arg))
    newres_2 = dcg.translate(rules, dcg.NonTerminal(category=cat, arg=r2.arg))
    if False and (newres_1 != oldres_1 or newres_2 != oldres_2):
        print('')
        for rule in rules:
            print(rule)
        print('')

        print('r1', r1)
        print('r2', r2)
        print('r1.arg', r1.arg)
        print('r2.arg', r2.arg)
        print('s_start [%s]' % s_start, 's_end [%s]' % s_end)
        print('s1_mid [%s]' % s1_mid, 's2_mid [%s]' % s2_mid)
        # print(rules)
        print('oldres 1 [%s]' % oldres_1)
        print('oldres 2 [%s]' % oldres_2)
        print('newres_1 [%s]' % newres_1)
        print('newres_2 [%s]' % newres_2)
    # assert newres_1 == oldres_1
    # assert newres_2 == oldres_2

    # print('new rules', rules)
    # import kirby2001
    # kirby2001.print_table(rules)
    return rules


def generalize(rules):
    ever_modified = False
    modified = True
    while modified:
        rules, modified = _generalize(rules)
        if modified:
            ever_modified = True
    return rules, ever_modified


def _generalize(rules):
    """
    input: rules
    output: simpler_rules, was_simplified
    """
    # was_simplified = False

    """
    Do we really have to do O(n^2) comparisons? :(
    I guess so :(
    I mean, we could choose not to, but I think kirby 2001 probably
    does them exhaustively

    to simplify the code, we're just going to do a single merge per pass...

    # note that if any rules are merged, those rules are both excluded from the rest of the pass
    # I'm not sure if this is a theoretical requirement, but certainly simplifies :)
    """
    # new_rules = []
    for i, r1 in enumerate(rules):
        # print('r1', r1)
        # merge_done = False
        for r2 in rules[i + 1:]:
            # print('r2', r2)
            new_rules = merge_categories(rules, r1, r2)
            if new_rules is not None:
                return new_rules, True
            new_rules = merge_chunked_bothrules(rules, r1, r2)
            if new_rules is not None:
                return new_rules, True
        #     res_new = None
        #     res_new = merge_categories(r1, r2)
        #     # if res_new is None:
        #         # 
        #     if res_new is not None:
        #         new_rules.append(res_new)
        #         merge_done = True
        #         break
        # if merge_done:
        #     continue
        # else:
        #     new_rules.append(r1)
        #     new_rules.append(r2)
    return rules, False
    # return simpler_rules, was_simplified
