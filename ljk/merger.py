import sys
import pandas as pd

def fix(target_stations, ref):
    res = {}
    for i in range(len(target_stations)):
        res[target_stations[i]] = target_stations[i]
        if target_stations[i].find('_') == -1: continue
        t = target_stations[i].split('_')[0]
        if t == 'miyun': 
            res[target_stations[i]] = 'miyun_aq'
        elif t == 'miyunshuik':
            res[target_stations[i]] = 'miyunshuiku_aq'
        else:
            for s in ref:
                if s[:len(t)] == t:
                    res[target_stations[i]] = s
                    break
    return res
def main():
    assert len(sys.argv) == 6, '<ans1> <score1> <ans2> <score2> <out>'
    _, ans1, s1, ans2, s2, o = sys.argv
    ans1 = pd.read_csv(ans1)
    ans2 = pd.read_csv(ans2)
    s1 = pd.read_csv(s1)
    s2 = pd.read_csv(s2)
    ans = ans1.copy()
    target_stations = list(set(map(lambda x: x.split('#')[0], ans.test_id)))
    stations = list(s1.stationId)
    print(target_stations, stations)
    fix_map = fix(target_stations, stations)
    for i in range(0, len(ans), 48):
        station = ans.iloc[i].test_id.split('#')[0]
        station = fix_map[station]
        assert ans.iloc[i].test_id[-2:] == '#0', ans.iloc[i].test_id
        is_ld = station[0].isupper()
        for j in ['PM2.5', 'PM10', 'O3']:
            print(station, j, s1[s1.stationId == station][j].iloc[0], s2[s2.stationId == station][j].iloc[0])
            assert len(s1[s1.stationId == station][j]) == 1, s1[s1.stationId == station][j]
            def check():
                if is_ld:
                    return s1[s1.stationId == station][j].iloc[0] > s2[s2.stationId == station][j].iloc[0] and j != 'O3'
                else:
                    return j == 'O3' or s1[s1.stationId == station][j].iloc[0] > s2[s2.stationId == station][j].iloc[0]
            if check():
                # print(ans2[ans2.test_id == ans.iloc[i].test_id])
                # print(ans2[ans2.test_id == ans.iloc[i].test_id][j])
                print('replace', ans.iloc[i][j], ans2[ans2.test_id == ans.iloc[i].test_id][j])
                # print(ans2[i:i+48][j])
                ans.iloc[i:i+48].loc[:, j] = ans2[i:i+48].loc[:, j].values
            else:
                print('skip')
    ans.to_csv(o, index=False)

if __name__ == '__main__':
    main()