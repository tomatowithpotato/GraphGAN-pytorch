import pandas as pd


class recommendation:
    def __init__(self):
        self.user_max = None
        self.movie_max = None
        self.watched = None
        self.unwatched = None
    
    def is_user(self, node):
        return node < self.user_max
    
    def read_csv_cols(self, dataframe, columns):
        result = []
        for col_name in columns:
            result.append(dataframe[col_name].values.tolist())
        return result

    def lists_sub_one(self, lists):
        lists = [[l-1 for l in lst] for lst in lists]
        return lists

    def read_edges_from_file(self, filename):
        u_m_ratings = pd.read_csv(filename)
        u_lst, m_lst = self.read_csv_cols(u_m_ratings, ['user_id','movie_id'])

        u_lst, m_lst = self.lists_sub_one([u_lst, m_lst]) # 编号调整为从0开始

        m_lst = [m + self.user_max for m in m_lst] # movie在矩阵中的id从usermax开始
        
        edges = list(zip(u_lst, m_lst))
        return edges

def get_umax_mmax(rcmd, train_filename, test_filename):
    train_ratings = pd.read_csv(train_filename)
    test_ratings = pd.read_csv(test_filename)
    train_u_lst, train_m_lst = rcmd.read_csv_cols(train_ratings, ['user_id', 'movie_id'])
    test_u_lst, test_m_lst = rcmd.read_csv_cols(test_ratings, ['user_id', 'movie_id'])
    user_max = max(train_u_lst + test_u_lst)
    movie_max = max(train_m_lst + test_m_lst)
    return user_max, movie_max


def read_edges(train_filename, test_filename):
    '''值得一提，在该推荐系统中，GAN的embedding matrix实际可看成两部分\n
    一部分是user，另一部分是movie\n
    为了防止id号冲突，因此movie在矩阵中的id == movie_id + user_max
    其中，user_max是最大的u_id。这样就可以把二者表示在同一个矩阵中了.
    '''
    rcmd = recommendation()

    user_max, movie_max = get_umax_mmax(rcmd, train_filename, test_filename)
    rcmd.user_max = user_max
    rcmd.movie_max = movie_max

    train_u_m_edges = rcmd.read_edges_from_file(train_filename)
    test_u_m_edges = rcmd.read_edges_from_file(test_filename)

    #construct user-movie graph
    graph = {}
    for u, m in train_u_m_edges:
        if u not in graph:
            graph[u] = []
        if m not in graph:
            graph[m] = []
        graph[u].append(m)
        graph[m].append(u)
            
    for u, m in test_u_m_edges:
        if u not in graph:
            graph[u] = []
        if m not in graph:
            graph[m] = []
    
    #测试集中用户实际观看(评分)过的电影
    watched = {}
    for u, m in test_u_m_edges:
        if u not in watched:
            watched[u] = set()
        watched[u].add(m)
    
    #训练集中未观看(评分)的电影
    unwatched = {}
    test_u_set = set([u for u,m in test_u_m_edges])
    for u in test_u_set:
        if u not in unwatched:
            unwatched[u] = []
        for m in range(user_max, user_max+movie_max):
            if m not in graph[u]:
                unwatched[u].append(m)
    
    rcmd.watched = watched
    rcmd.unwatched = unwatched

    #print(len(graph.keys()), min(list(graph.keys())), max(list(graph.keys())), user_max)
    return graph, rcmd
