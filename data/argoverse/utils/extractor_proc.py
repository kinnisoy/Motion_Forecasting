from scipy import sparse
import pandas as pd
import numpy as np
from pathlib import Path
from av2.map.map_api import ArgoverseStaticMap


class ArgoDataExtractor:
    def __init__(self, args):
        self.align_image_with_target_x = args.align_image_with_target_x
        self.num_scales = args.num_scales
        self.cross_dist = 6

    def get_displ(self, data):
        """
        Get x and y displacements (proportional to discrete velocities) for
        a given trajectory and update the valid flag for observed timesteps
        获取位移
        Args:
            data: Trajectories of all agents
        Returns:
            Displacements of all agents
        """
        res = np.zeros((data.shape[0], data.shape[1] - 1, data.shape[2]))

        for i in range(len(res)):
            # Replace  0 in first dimension with 2
            # 将第一维的0替换为2 ？？？   不就是后一个坐标减去前一个坐标的偏移 shift
            diff = data[i, 1:, :2] - data[i, :-1, :2]
            
            """ we only consider vehicles that are observable at t = 0 and handle vehicles that are not
                observed over the full history horizon Th by concatenating a binary flag b.The flag indicates whether there was a
                displacement of vehicle i observed at timestep t
                只考虑 t = 0 时候的车辆，其他整个轨迹中观察不到的车辆通过连接一个flag，表明其是否发生偏移。
                """
            # Sliding window (size=2) with the valid flag , linear convolution of two one-dimensional sequences
            # 带有有效标志的滑动窗口（大小=2），两个一维序列的线性卷积核
            ''' data[i, :, 2]是标记位flag  
                ones（2）= [1,1] 是卷积核
                valid 线性卷积模式  返回完全重叠的值
            '''
            valid = np.convolve(data[i, :, 2], np.ones(2), "valid")
            # Valid entries have the sum=2 (they get flag=1=valid), unvalid entries have the sum=1 or sum=2 (they get flag=0)
            # 挑选出所有有效的点 卷积核内两个都是1和为2 的点，表示相邻的两个点都是被观测到的。
            valid = np.select(
                [valid == 2, valid == 1, valid == 0], [1, 0, 0], valid)

            res[i, :, :2] = diff
            res[i, :, 2] = valid

            # Set zeroes everywhere, where third dimension is = 0 (invalid)
            res[i, res[i, :, 2] == 0] = 0

        return np.float32(res), data[:, -1, :2]

    def extract_data(self, filename,map_file):
        """
        Load parquet and extract the features required for TFMF (Trsformers for Motion Forecasting)

        Args:
            filename: Filename of the parquet to load

        Returns:
            Feature dictionary required for TFMF
        """

        df = pd.read_parquet(filename)
        argo_id = Path(filename).stem.split('_')[-1]
        # print(df[['position_x','position_y','timestep']][df['track_id']=='72146'])
       
        city = df["city"].values[0]
        track_id = df["track_id"].values[0]
     
        agt_ts = np.sort(np.unique(df["timestep"].values))
     

        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i
        
     

        trajs = np.concatenate((
            df.position_x.to_numpy().reshape(-1, 1),
            df.position_y.to_numpy().reshape(-1, 1)), 1)
        

        steps = [mapping[x] for x in df["timestep"].values]
        steps = np.asarray(steps, np.int64)

        # replace focal_track_id and AV in object_type
        df['object_type']= df.apply(lambda row: 'AGENT' if row['track_id']==row['focal_track_id'] else row['object_type'],axis=1)
        df['object_type']= df.apply(lambda row: 'AV' if row['track_id']=='AV' else row['object_type'],axis=1)

        objs = df.groupby(["track_id", "object_type"]).groups
        keys = list(objs.keys())
       
        obj_type = [x[1] for x in keys]
    
      
        agnt_key = keys.pop(obj_type.index("AGENT"))
        av_key = keys.pop(obj_type.index("AV")-1)
        keys = [agnt_key, av_key] + keys

        res_trajs = []
        for key in keys:
            idcs = objs[key]    
            tt = trajs[idcs]
            ts = steps[idcs]
            rt = np.zeros((110, 3))

            if 49 not in ts:
                continue

            rt[ts, :2] = tt
            rt[ts, 2] = 1.0  # the flag columns of each agent at time steps where the agent is observed is considered 1
            res_trajs.append(rt)


        res_trajs = np.asarray(res_trajs, np.float32)
        res_gt = res_trajs[:, 50:].copy()
        origin = res_trajs[0, 49, :2].copy().astype(np.float32)
        """ 
        During preprocessing, coordinate transformation of each sequence into a local target vehicle coordinate frame is done. 
        This common preprocessing step is also performed by other approaches [3], [25] benchmarked on the Argoverse dataset. 
        Therefore, the coordinates in each sequence are transformed into a coordinate frame originated at the position of the target vehicle at t = 0.
         The orientation of the positive x-axis is given by the vector described by the difference between the position at t = 0 and t = −1.
        将每个序列中的坐标转换成源自t＝0处目标车辆位置的坐标系。
        正x轴的方向由t=0和t=−1位置之间的差所描述的矢量给出
        """

        # The eye tool returns a 2-D array with  1’s as the diagonal and  0’s elsewhere.
        # 对角线矩阵
        rotation = np.eye(2, dtype=np.float32)

        theta = 0

        # 根据目标的x坐标对齐图像
        if self.align_image_with_target_x:
            pre = res_trajs[0, 49, :2] - res_trajs[0, 48, :2]
            theta = np.arctan2(pre[1], pre[0])
            rotation = np.asarray([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]], np.float32)

        res_trajs[:, :, :2] = np.dot(res_trajs[:, :, :2] - origin, rotation) #Dot product of two arrays
        res_trajs[np.where(res_trajs[:, :, 2] == 0)] = 0

        res_fut_trajs = res_trajs[:, 50:].copy()
        res_trajs = res_trajs[:, :50].copy()

      
        sample = dict()
        sample["argo_id"] = argo_id
        sample["city"] = city
        sample["past_trajs"] = res_trajs
        sample["fut_trajs"] = res_fut_trajs
        # GROUND Truth是坐标未转换之前的
        sample["gt"] = res_gt[:, :, :2]
        sample["displ"], sample["centers"] = self.get_displ(sample["past_trajs"])
        sample["origin"] = origin
        # We already return the inverse transformation matrix, Compute the (multiplicative) inverse of a matrix
        sample["rotation"] = np.linalg.inv(rotation)   # 求逆
        sample["graph"] = self.extract_map(map_file,rotation,origin)

        return sample

    def extract_map(self,filename,rotation,origin):
        filename = Path(filename)
        # Note : read json file , Not from  DIR
        # avm = ArgoverseStaticMap.from_map_dir(filename, build_raster=False)
        """Get all lane within this scene, convert centerline and polygon to rotated and biased"""
        avm = ArgoverseStaticMap.from_json(filename)
        lane_ids = avm.get_scenario_lane_segment_ids()
        lanes = avm.get_scenario_lane_segments()
        # ctrs, sucs, lefts, rights, pres, feats, intersection = [], [], [], [], [], [], []
        ctrs, feats, intersection= [],[],[]
        for lane_id in lane_ids:
            """Lane feature: ctrs(position), feats(shape), nbrs, intersect"""
            """Note that every lane has different points or nodes in it."""
            ctr = avm.get_lane_segment_centerline(lane_segment_id=lane_id)
            ctr = np.matmul(rotation,(ctr[:,:2]-origin.reshape(-1,2)).T).T
            # suc = avm.get_lane_segment_successor_ids(lane_segment_id=lane_id)
            # left = avm.get_lane_segment_left_neighbor_id(lane_segment_id=lane_id)
            # right = avm.get_lane_segment_right_neighbor_id(lane_segment_id=lane_id)
            # pre = list(reversed(suc))
            intersect = avm.lane_is_in_intersection(lane_segment_id=lane_id)

            ctrs.append(np.asarray((ctr[:-1] + ctr[1:]) / 2.0, np.float32))  # 前一个点+后一个点 除以2  中心的中心？
            feats.append(np.asarray(ctr[1:] - ctr[:-1], np.float32))
            # sucs.append(suc)
            # lefts.append(left)
            # rights.append(right)
            # pres.append(pre)
            if intersect:
                intersection.append(np.ones((1,)))
            else:
                intersection.append(np.zeros((1,)))
        ''' num_nodes'''
        count = 0
        node_idcs = []
        for i, ctr in enumerate(ctrs):  # node_idcs: list, i-th element: i-th lane nodes ids
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count
        # -------------------------lane_idcs---------------------
        # lane[idc] = a means idc-th node belongs to the a-th lane
        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)
        """
        # **********************************Map Related work***************************
        # =========================================
        # ==============Hdmap Graph Build==========
        # =========================================
        # ---------------------------pre and suc for lanes--------------------
        """

        '''paris * 4 '''
        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[i]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.left_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.right_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)


        graph = dict()

        graph['lane_idcs'] = lane_idcs
        graph['intersection']= np.concatenate(intersection,0)
        graph['ctrs'] = np.concatenate(ctrs,0)
        graph['feats'] = np.concatenate(feats,0)
        graph['lane_ids'] = lane_ids
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs

        # ---------------------------pre and suc for nodes--------------------
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids): # for every lane in this map.
            lane = lanes[i]       # take out this lane.
            idcs = node_idcs[i]   # whole nodes index in this lane.

            pre['u'] += idcs[1:]   # Set pre['u'][0] = idcs[1:]
            pre['v'] += idcs[:-1]  # pre refers that from V SET can arrive U SET. U refers Destination, and V refers Source nodes set.
            # - pre['u'] , pre['u'] contains the destination nodes.
            # for example : pre['u'][0] contains all nodes without the first one in this lane, means the current node's destination can be these nodes.
            # - As the same, pre['v'] contains the source nodes.
            if lane.predecessors is not None:    # predecessor 前置 走过的路
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids: # if the pre nbr node in this scene , In Argovers 2.0, it seems not to consider.
                        j = lane_ids.index(nbr_id)  # take out this nbr's index
                        pre['u'].append(idcs[0])    # the destination node is the first node of this lane.
                        pre['v'].append(node_idcs[j][-1]) # the start node is last node of pre lane.| v is the pre of u, v is src, u is dest.

            suc['u'] += idcs[:-1]   # destination. all nodes without last one can arrive here [last node].
            suc['v'] += idcs[1:]    # source.
            if lane.successors is not None:   # 后面要走的路
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])   # Todo: what means?
                        suc['v'].append(node_idcs[j][0])

        pre['u'] = np.asarray(pre['u'], dtype=np.int16)
        pre['v'] = np.asarray(pre['v'], dtype=np.int16)
        suc['u'] = np.asarray(suc['u'], dtype=np.int16)
        suc['v'] = np.asarray(suc['v'], dtype=np.int16)

        # -------------------dilate pre and suc: opition 1--------------------
        dilated_pre = [pre]
        dilated_pre += self.dilated_nbrs(pre, num_nodes, self.num_scales)
        dilated_suc = [suc]
        dilated_suc += self.dilated_nbrs(suc, num_nodes, self.num_scales)

        # --------------------build nodes left and right graph-----------------
        num_lanes = lane_idcs[-1].item() + 1

        left, right = dict(), dict()
            # (810,1,3)                                        (1,810,3)
        dist = np.expand_dims(graph['ctrs'], axis=1) - np.expand_dims(graph['ctrs'], axis=0)
        dist = np.sqrt((dist ** 2).sum(2))
        hi = np.arange(num_nodes).reshape(-1, 1).repeat(num_nodes, axis=1).reshape(-1)  #72900 = 270**2 axis=1 expand by h columns
        wi = np.arange(num_nodes).reshape(1, -1).repeat(num_nodes, axis=0).reshape(-1)
        row_idcs = np.arange(num_nodes)

        pre_mat = np.zeros((num_lanes, num_lanes), dtype=float)
        if len(pre_pairs) > 0: # if pre_pairs is empty, pre_pairs[:, 0] will raise Error
            # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
            pre_mat[pre_pairs[:, 0], pre_pairs[:, 1]] = 1
        suc_mat = np.zeros((num_lanes, num_lanes), dtype=float)
        if len(suc_pairs) > 0:
            suc_mat[suc_pairs[:, 0], suc_pairs[:, 1]] = 1

        pairs = left_pairs
        if len(pairs) > 0:
            # construct lane left graph
            mat = np.zeros((num_lanes, num_lanes))
            mat[pairs[:, 0], pairs[:, 1]] = 1
            mat = (np.matmul(mat, pre_mat) + np.matmul(mat,
                                                       suc_mat) + mat) > 0.5  # left lane's suc or pre lane is also self's left lane

            # filter with distance
            left_dist = dist.copy()
            # if lane j is the lane i's left, then all nodes in lane j is the left of any node in lane i
            mask = np.logical_not(mat[lane_idcs[hi], lane_idcs[wi]])
            # set the distance between nodes that has no left relation are very vert large
            left_dist[hi[mask], wi[mask]] = 1e6

            # find the each node's nearest node
            min_dist, min_idcs = left_dist.min(1), left_dist.argmin(1)
            # if nearest node's distance > self.config['cross_dist'], then this node does not have left node
            mask = min_dist < self.cross_dist
            # if the angle between nearest node is too big , the this node does not have left node
            ui = row_idcs[mask]
            vi = min_idcs[mask]
            f1 = graph['feats'][ui]
            f2 = graph['feats'][vi]
            t1 = np.arctan2(f1[:, 1], f1[:, 0])
            t2 = np.arctan2(f2[:, 1], f2[:, 0])
            dt = np.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = np.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            left['u'] = ui.astype(np.int16)  # u is the idx of node that has left neighbor
            left['v'] = vi.astype(np.int16)  # v[i] is the idx of left neighbor of node u[i]
        else:
            left['u'] = np.zeros(0, np.int16)
            left['v'] = np.zeros(0, np.int16)

        pairs = right_pairs
        if len(pairs) > 0:
            mat = np.zeros((num_lanes, num_lanes), dtype=float)
            mat[pairs[:, 0], pairs[:, 1]] = 1
            mat = (np.matmul(mat, pre_mat) + np.matmul(mat, suc_mat) + mat) > 0.5

            right_dist = dist.copy()
            mask = np.logical_not(mat[lane_idcs[hi], lane_idcs[wi]])
            right_dist[hi[mask], wi[mask]] = 1e6

            min_dist, min_idcs = right_dist.min(1), right_dist.argmin(1)
            mask = min_dist < self.cross_dist
            ui = row_idcs[mask]
            vi = min_idcs[mask]
            f1 = graph['feats'][ui]
            f2 = graph['feats'][vi]
            t1 = np.arctan2(f1[:, 1], f1[:, 0])
            t2 = np.arctan2(f2[:, 1], f2[:, 0])
            dt = np.abs(t1 - t2)
            m = dt > np.pi
            dt[m] = np.abs(dt[m] - 2 * np.pi)
            m = dt < 0.25 * np.pi

            ui = ui[m]
            vi = vi[m]

            right['u'] = ui.astype(np.int16)
            right['v'] = vi.astype(np.int16)
        else:
            right['u'] = np.zeros(0, np.int16)
            right['v'] = np.zeros(0, np.int16)


        '''generate_graph'''
        graph['pre'] = dilated_pre
        graph['suc'] = dilated_suc
        graph['left'] = left
        graph['right'] = right


        return graph

    def dilated_nbrs(self, nbr, num_nodes, num_scales):
        data = np.ones(len(nbr['u']), np.bool)
        csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

        mat = csr
        nbrs = []
        for i in range(1, num_scales):
            mat = mat * mat

            nbr = dict()
            coo = mat.tocoo()
            nbr['u'] = coo.row.astype(np.int16)
            nbr['v'] = coo.col.astype(np.int16)
            nbrs.append(nbr)
        return nbrs