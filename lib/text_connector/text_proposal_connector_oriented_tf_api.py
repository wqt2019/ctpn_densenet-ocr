#coding:utf-8
# import numpy as np
# from .text_proposal_graph_builder import TextProposalGraphBuilder
from .text_connect_cfg import Config as TextLineCfg

#---------------------
import tensorflow as tf


class TextProposalConnector:


    #--------tf-api实现--------------
    def get_text_lines_tf_api(self, text_proposals_tf, scores_tf, im_info_tf):

        # 首先还是建图，获取到文本行由哪几个小框构成
        # 组合相邻的text_proposals，successions_pair为配对结果，sub_graphs_pair为每组最右边的text_proposals
        successions_pair, sub_graphs_pair=self.group_text_proposals_tf_api(text_proposals_tf, scores_tf, im_info_tf)

        # ------------------------
        def my_cond_text_lines(loop_i,  tmp_tf_input):
            return tf.less(loop_i, loop_len)

        def my_body_text_lines(loop_i, tmp_tf_input):

            tp_indices_tf = self.final_group_tf_api(successions_pair, sub_graphs_pair, loop_i)
            text_line_boxes_tf = tf.gather(text_proposals_tf,tp_indices_tf)

            X_tf = (text_line_boxes_tf[:, 0] + text_line_boxes_tf[:, 2]) / 2  # 求每一个小框的中心x，y坐标
            Y_tf = (text_line_boxes_tf[:, 1] + text_line_boxes_tf[:, 3]) / 2

            X_tf = tf.reshape(X_tf,[-1,1])
            ones = tf.ones(shape=tf.shape(X_tf))
            ones = tf.cast(ones,tf.float32)

            X_tf = tf.concat([X_tf, ones], 1)
            Y_tf = tf.reshape(Y_tf,[-1,1])
            z1 = tf.matrix_solve_ls(X_tf, Y_tf) #求解多个线性方程的最小二乘问题 ---- np.polyfit(X, Y, 1)

            x0_index = tf.argmin(text_line_boxes_tf[:, 0])  # 文本行x坐标最小值
            x1_index = tf.argmax(text_line_boxes_tf[:, 2])  # 文本行x坐标最大值
            x0 = tf.gather(text_line_boxes_tf[:, 0],x0_index)
            x1 = tf.gather(text_line_boxes_tf[:, 2], x1_index)
            offset = (text_line_boxes_tf[0, 2] - text_line_boxes_tf[0, 0]) * 0.5  # 小框宽度的一半

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lt_y, rt_y = self.fit_y_tf_api(text_line_boxes_tf[:, 0], text_line_boxes_tf[:, 1], x0 + offset, x1 - offset)
            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lb_y, rb_y = self.fit_y_tf_api(text_line_boxes_tf[:, 0], text_line_boxes_tf[:, 3], x0 + offset, x1 - offset)

            tmp_scores=tf.gather(scores_tf, tp_indices_tf)
            score = tf.reduce_mean(tmp_scores)  # 求全部小框得分的均值作为文本行的均值

            height = tf.reduce_mean((text_line_boxes_tf[:, 3] - text_line_boxes_tf[:, 1]))  # 小框平均高度
            height = tf.add(height,2.5)

            x0=tf.reshape(x0, [-1])
            ty=tf.reshape(tf.minimum(lt_y, rt_y), [-1]) # 文本行上端 线段 的y坐标的小值
            x1=tf.reshape(x1, [-1])
            by=tf.reshape(tf.maximum(lb_y, rb_y), [-1]) # 文本行下端 线段 的y坐标的大值
            score=tf.reshape(score, [-1]) # 文本行得分
            z1_0=tf.reshape(z1[0],[-1]) # 根据中心点拟合的直线的k，b
            z1_1=tf.reshape(z1[1],[-1])
            height=tf.reshape(height,[-1])

            text_lines_tf_tmp  = tf.concat([x0,ty,x1,by,score,z1_0,z1_1,height],0)
            result = tf.concat([tmp_tf_input,tf.reshape(text_lines_tf_tmp,shape=[1,8])],0)

            return loop_i + 1,  result

        loop_len = tf.shape(sub_graphs_pair)[0]
        i = tf.constant(0)
        tmp_tf = tf.zeros((1, 8), tf.float32)
        _, text_lines_tf = tf.while_loop(cond=my_cond_text_lines, body=my_body_text_lines,
                            loop_vars=[i, tmp_tf],
                            shape_invariants=[tf.TensorShape(None), tf.TensorShape([None, 8])])

        tmp_text_lines_tf=text_lines_tf[1:]

        # ------------------------
        def my_cond(loop_i,tmp_text_lines_tf_input,tmp_tf_input):
            return tf.less(loop_i, loop_len)

        def my_body(loop_i,tmp_text_lines_tf_input,tmp_tf_input):
            line_tf = tmp_text_lines_tf_input[loop_i]
            b1 = line_tf[6] - line_tf[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line_tf[6] + line_tf[7] / 2
            x1 = line_tf[0]
            y1 = line_tf[5] * line_tf[0] + b1  # 左上
            x2 = line_tf[2]
            y2 = line_tf[5] * line_tf[2] + b1  # 右上
            x3 = line_tf[0]
            y3 = line_tf[5] * line_tf[0] + b2  # 左下
            x4 = line_tf[2]
            y4 = line_tf[5] * line_tf[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = tf.sqrt(disX * disX + disY * disY)  # 文本行宽度
            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = tf.abs(fTmp1 * disX / width)  # 做补偿
            y = tf.abs(fTmp1 * disY / width)

            line_5 = line_tf[5]  # 替换numpy中的   if--else
            less_fn = lambda: (x1 - x, y1 + y, x2, y2, x3, y3, x4 + x, y4 - y)
            more_fn = lambda: (x1, y1, x2 + x, y2 + y, x3 - x, y3 - y, x4, y4)
            x1, y1, x2, y2, x3, y3, x4, y4 = tf.cond(line_5 < 0.00, true_fn=less_fn, false_fn=more_fn)

            x1 = tf.reshape(x1, [-1])
            y1 = tf.reshape(y1, [-1])
            x2 = tf.reshape(x2, [-1])
            y2 = tf.reshape(y2, [-1])
            x3 = tf.reshape(x3, [-1])
            y3 = tf.reshape(y3, [-1])
            x4 = tf.reshape(x4, [-1])
            y4 = tf.reshape(y4, [-1])
            line_4 = tf.reshape(line_tf[4], [-1])

            text_recs_tf_tmp = tf.concat([x1, y1, x2, y2, x3, y3, x4, y4, line_4], 0)
            result = tf.concat([tmp_tf_input, tf.reshape(text_recs_tf_tmp, shape=[1, 9])], 0)

            return loop_i+1,tmp_text_lines_tf_input,result

        loop_len=tf.shape(tmp_text_lines_tf)[0]
        i=tf.constant(0)
        tmp_tf = tf.zeros((1, 9), tf.float32)
        _,_,text_recs_tf = tf.while_loop(cond=my_cond, body=my_body, loop_vars=[i,tmp_text_lines_tf,tmp_tf],
                         shape_invariants=[tf.TensorShape(None), tf.TensorShape([None, 8]),tf.TensorShape([None, 9])])

        result_recs=text_recs_tf[1:]

        #----10*10以下的box忽略-----
        keep = tf.where((result_recs[:,0] < result_recs[:,6] - 10) & (result_recs[:,1] < result_recs[:,7] - 10))[:,0]
        result_recs1 = tf.gather(result_recs, keep)

        return result_recs1

    #直线拟合
    def fit_y_tf_api(self, X, Y, x1, x2):
        # len(X)!=0
        # # if X only include one point, the function will get line y=Y[0]
        # if np.sum(X==X[0])==len(X):
        #     return Y[0], Y[0]

        X_tf = tf.reshape(X, [-1, 1])
        ones = tf.ones(shape=tf.shape(X_tf))
        ones = tf.cast(ones,tf.float32)
        X_tf = tf.concat([X_tf, ones], 1)
        Y_tf = tf.reshape(Y, [-1, 1])
        z1 = tf.matrix_solve_ls(X_tf, Y_tf)
        k=z1[0]
        b=z1[1]
        p1 = tf.add(tf.multiply(k, x1), b)
        p2 = tf.add(tf.multiply(k, x2), b)

        return p1,p2


    #对单个proposals进行配对，返回配对的结果在text_proposals_tf中的序号
    def get_successions_tf_api(self, index):
        box = self.text_proposals_tf_int[index]
        a = box[0]
        b = tf.minimum(box[0] + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_info_tf[1]) #不超过边界
        adj_box_indices=tf.where((self.text_proposals_tf_int[:, 0] > a) & (self.text_proposals_tf_int[:, 0] < b))
        adj_box_indices=tf.cast(adj_box_indices, tf.int32)
        adj_box_indices_len=tf.shape(adj_box_indices)[0]

        def cond(loop_i,adj_box_indices_input,heights_tf_input,text_proposals_tf_input,tmp_tf_input):
            return tf.less(loop_i, adj_box_indices_len)

        def body(loop_i,adj_box_indices_input,heights_tf_input,text_proposals_tf_input,tmp_tf_input):
            cur_adj_box_indices = adj_box_indices_input[loop_i][0]
            h1 = heights_tf_input[index]
            h1 = tf.cast(h1, tf.float32)
            h2 = heights_tf_input[cur_adj_box_indices]
            h2 = tf.cast(h2, tf.float32)
            y0 = tf.maximum(text_proposals_tf_input[cur_adj_box_indices][1], text_proposals_tf_input[index][1])
            y0 = tf.cast(y0, tf.float32)
            y1 = tf.minimum(text_proposals_tf_input[cur_adj_box_indices][3], text_proposals_tf_input[index][3])
            y1 = tf.cast(y1, tf.float32)

            #计算meet_v_iou
            a1=tf.divide(tf.maximum(0.00, y1 - y0 + 1), tf.minimum(h1, h2))
            a1=tf.reshape(a1,[1])
            #计算size_similarity
            a2=tf.divide(tf.minimum(h1, h2), tf.maximum(h1, h2))
            a2 = tf.reshape(a2, [1])


            text_recs_tf_tmp = tf.concat([a1, a2], 0)
            result = tf.concat([tmp_tf_input, tf.reshape(text_recs_tf_tmp, shape=[1, 2])], 0)

            return loop_i+1,adj_box_indices_input,heights_tf_input,text_proposals_tf_input,result

        i = tf.constant(0)
        tmp_tf = tf.zeros((1, 2), tf.float32)
        self.heights_tf=tf.reshape(self.heights_tf, [-1])
        #对每一个box进行配对
        _, _, _,_,text_box_indices = tf.while_loop(cond=cond, body=body,
                                loop_vars=[i, adj_box_indices, self.heights_tf, self.text_proposals_tf, tmp_tf],
                                    shape_invariants=[tf.TensorShape(None), tf.TensorShape([None,1]),
                                    tf.TensorShape(None),tf.TensorShape([None,4]),tf.TensorShape([None,2])])

        text_box_indices_tmp=text_box_indices[1:]
        indices_tmp = tf.where((text_box_indices_tmp[:,0]>TextLineCfg.MIN_V_OVERLAPS) &
                               (text_box_indices_tmp[:,1]>TextLineCfg.MIN_SIZE_SIM))
        indices_tmp_shape=tf.shape(indices_tmp)[0]

        #判断indices_tmp 是否为空 , 为空就赋值 -1
        def f1():
            # 可能有多个配对，取最近的一个
            tmp_i = tf.gather(adj_box_indices, indices_tmp)
            tmp_i1 = tf.reshape(tmp_i, [-1])
            tmp_x0 = tf.gather(self.text_proposals_tf_int[:, 0], tmp_i1) - a
            tmp_min_index = tf.argmin(tmp_x0)  #取距离最近的
            tmp_ii = indices_tmp[tmp_min_index]
            tmp_ii = tmp_ii[0]
            aa1 = adj_box_indices[tmp_ii]
            # aa1 = tf.cast(aa1, tf.float32)
            return aa1

        def f2():
            aa2 = tf.constant(-1, shape=[1])
            # aa2 = tf.cast(aa2,tf.float32)
            return aa2

        result_text_box_indices = tf.cond(tf.equal(indices_tmp_shape,0), f2, f1)

        return result_text_box_indices


    #对多有的text_proposals进行配对
    def group_text_proposals_tf_api(self, text_proposals_tf, scores_tf, im_info_tf):

        im_info_tf = tf.cast(im_info_tf, tf.int32)
        self.text_proposals_tf_int = tf.cast(text_proposals_tf, tf.int32)
        self.text_proposals_tf = text_proposals_tf
        self.scores_tf = scores_tf
        self.im_info_tf = im_info_tf
        self.heights_tf = text_proposals_tf[:, 3] - text_proposals_tf[:, 1] + 1

        #debug
        # i=tf.constant(150)  #text_proposals_tf中的序号
        # a=self.get_successions_tf_api(i)


        #对所有的text_proposals_tf进行配对
        text_proposals_len=tf.shape(text_proposals_tf)[0]
        def my_cond(loop_i,tmp_tf_input):
            return tf.less(loop_i, text_proposals_len)

        def my_body(loop_i,tmp_tf_input):
            #对单个text_proposals进行配对， loop_i--successions_index
            successions_index=self.get_successions_tf_api(loop_i)
            loop_i_1=tf.reshape(loop_i, [1])
            successions_index = tf.reshape(successions_index, [1])
            text_recs_tf_tmp = tf.concat([loop_i_1, successions_index], 0)
            result = tf.concat([tmp_tf_input, tf.reshape(text_recs_tf_tmp, shape=[1, 2])], 0)

            return loop_i+1,result

        index = tf.constant(0)
        tmp_tf = tf.zeros((1, 2), tf.int32)
        _,  successions_pair = tf.while_loop(cond=my_cond, body=my_body, loop_vars=[index, tmp_tf],
                                    shape_invariants=[tf.TensorShape(None), tf.TensorShape([None, 2])])

        result_successions_pair=successions_pair[1:]  #所有的text_proposals_tf配对的结果
        result_successions_pair=tf.cast(result_successions_pair,tf.int32)

        sub_graphs_pair = tf.where(tf.equal(result_successions_pair[:,1],-1))
        sub_graphs_pair = tf.cast(sub_graphs_pair,tf.int32)

        return result_successions_pair,sub_graphs_pair


    def final_group_tf_api(self, result_successions_pair, sub_graphs_pair, cur_index):
    #对应源码中的 sub_graphs_connected(self)函数
        def my_cond_1(loop_i,sub_graphs_pair_input,tmp_tf_input):
            cur_ii=tf.where(tf.equal(result_successions_pair[:,1],loop_i))
            indices_tmp_shape = tf.shape(cur_ii)[0]
            a = tf.cond(tf.equal(indices_tmp_shape, 0), lambda: False, lambda: True)
            return a

        def my_body_1(loop_i,sub_graphs_pair_input,tmp_tf_input):
            cur_iii = tf.where(tf.equal(result_successions_pair[:, 1], loop_i))
            cur_iii = tf.cast(cur_iii,tf.int32)
            cur_iii = cur_iii[0]  #可能有多个配对，取第一个
            cur_iii = tf.reshape(cur_iii, shape=[1])
            result = tf.concat([tmp_tf_input, cur_iii ], 0)

            return cur_iii,sub_graphs_pair_input,result

        index=sub_graphs_pair[cur_index]
        index = tf.reshape(index,shape=[1])
        tmp_tf = tf.constant(0,shape=[1])
        tmp_tf = tf.concat([tmp_tf,index],0)

        _, _ , result_group = tf.while_loop(cond=my_cond_1, body=my_body_1, loop_vars=[index,sub_graphs_pair, tmp_tf],
                                    shape_invariants=[tf.TensorShape(None), tf.TensorShape(None),tf.TensorShape(None)])

        final_result_group = result_group[1:]
        return final_result_group
