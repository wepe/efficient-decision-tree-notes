package main;

import java.util.*;
import java.util.concurrent.*;

public class TreeBack {
    public TreeNode root;
    public int min_sample_split;
    public double min_child_weight;
    public int max_depth;
    public double colsample;
    public double rowsample;
    public double lambda;
    public double gamma;
    public int num_thread;
    public Queue<TreeNode> alive_nodes = new LinkedList<>();
    //number of tree node of this tree
    public int nodes_cnt = 0;
    //number of nan tree node of this tree
    public int nan_nodes_cnt = 0;


    public TreeBack(int min_sample_split,
                double min_child_weight,
                int max_depth,
                double colsample,
                double rowsample,
                double lambda,
                double gamma,
                int num_thread){
        this.min_sample_split = min_sample_split;
        this.min_child_weight = min_child_weight;
        this.max_depth = max_depth;
        this.colsample = colsample;
        this.rowsample = rowsample;
        this.lambda = lambda;
        this.gamma = gamma;

        if(num_thread==-1){
            this.num_thread = Runtime.getRuntime().availableProcessors();
        }else {
            this.num_thread = num_thread;
        }
        //to avoid divide zero
        this.lambda = Math.max(this.lambda, 0.00001);
    }

    public double calculate_leaf_score(double G,double H){
        //According to xgboost, the leaf score is : - G / (H+lambda)
        return -G/(H+this.lambda);
    }

    public double[] calculate_split_gain(double G_left,double H_left,double G_nan,double H_nan,double G_total,double H_total){
        //According to xgboost, the scoring function is:
        //     gain = 0.5 * (GL^2/(HL+lambda) + GR^2/(HR+lambda) - (GL+GR)^2/(HL+HR+lambda)) - gamma
        //this gain is the loss reduction, We want it to be as large as possible.
        double G_right = G_total - G_left - G_nan;
        double H_right = H_total - H_left - H_nan;

        //if we let those with missing value go to a nan child
        double gain_1 = 0.5 * (
                Math.pow(G_left,2)/(H_left+lambda)
                        + Math.pow(G_right,2)/(H_right+lambda)
                        + Math.pow(G_nan,2)/(H_nan+lambda)
                        - Math.pow(G_total,2)/(H_total+lambda));

        //if we let those with missing value go to left child
        double gain_2 = 0.5 * (
                Math.pow(G_left+G_nan,2)/(H_left+H_nan+lambda)
                        + Math.pow(G_right,2)/(H_right+lambda)
                        - Math.pow(G_total,2)/(H_total+lambda));

        //if we let those with missing value go to right child
        double gain_3 = 0.5 * (
                Math.pow(G_left,2)/(H_left+lambda)
                        + Math.pow(G_right+G_nan,2)/(H_right+H_nan+lambda)
                        - Math.pow(G_total,2)/(H_total+lambda));

        double nan_go_to;
        double gain = Math.max(gain_1,Math.max(gain_2,gain_3));
        if(gain_1==gain){
            nan_go_to = 0; //nan child
        }else if(gain_2==gain){
            nan_go_to = 1; //left child
        }else{
            nan_go_to = 2; //right child
        }

        //in this case, the trainset does not contains nan samples
        if(H_nan==0 && G_nan==0){
            nan_go_to = 3;
        }

        return new double[]{nan_go_to,gain};
    }


    public void fit(AttributeList attribute_list,
                    ClassList class_list,
                    RowSampler row_sampler,
                    ColumnSampler col_sampler){
        //when we start to fit a tree, we first conduct row and column sampling
        col_sampler.shuffle();
        row_sampler.shuffle();
        class_list.sampling(row_sampler.row_mask);


        //then we create the root node, initialize histogram(Gradient sum and Hessian sum)
        TreeNode root_node = new TreeNode(1,1,attribute_list.feature_dim,false);
        root_node.Grad_setter(sum(class_list.grad));
        root_node.Hess_setter(sum(class_list.hess));
        this.root = root_node;


        //put it into the alive_node, and fill the class_list, all data are assigned to root node initially
        alive_nodes.offer(root_node);


        for(int i=0;i<class_list.dataset_size;i++){
            class_list.corresponding_tree_node[i] = root_node;
        }


        //update Grad_missing Hess_missing for root node
        class_list.update_grad_hess_missing_for_tree_node(attribute_list.missing_value_attribute_list);

        //then build the tree util there is no alive tree_node to split
        build(attribute_list,class_list,col_sampler);
        clean_up();
    }

    class ProcessEachAttributeList implements Runnable{
        public int col;
        public AttributeList attribute_list;
        public ClassList class_list;
        public ProcessEachAttributeList(int col,AttributeList attribute_list,ClassList class_list){
            this.col = col;
            this.attribute_list = attribute_list;
            this.class_list = class_list;
        }

        @Override
        public void run(){
            for(int interval=0;interval<attribute_list.cutting_inds[col].length-1;interval++){
                //update the corresponding treenode's G_left,H_left with this inds's sample
                int[] inds = attribute_list.cutting_inds[col][interval];

                HashSet<TreeNode> nodes = new HashSet<>();
                for(int ind:inds){
                    TreeNode treenode = class_list.corresponding_tree_node[ind];
                    nodes.add(treenode);


                    treenode.G_left[col] += class_list.grad[ind];
                    treenode.H_left[col] += class_list.hess[ind];
                }
                //update each treenode's best split using this feature
                for(TreeNode node:nodes){
                    double G_left = node.G_left[col];
                    double H_left = node.H_left[col];
                    double G_total = node.Grad;
                    double H_total = node.Hess;
                    double G_nan = node.Grad_missing[col];
                    double H_nan = node.Hess_missing[col];
                    double[] ret = calculate_split_gain(G_left,H_left,G_nan,H_nan,G_total,H_total);
                    double nan_go_to = ret[0];
                    double gain = ret[1];
                    node.update_best_split(col,attribute_list.cutting_thresholds[col][interval],gain,nan_go_to);
                }
            }

        }
    }


    public void build(AttributeList attribute_list,
                      ClassList class_list,
                      ColumnSampler col_sampler){
        while(!alive_nodes.isEmpty()){
            nodes_cnt += alive_nodes.size();

            //parallelly scan and process each selected attribute list
            ExecutorService pool = Executors.newFixedThreadPool(num_thread);
            for(int col:col_sampler.col_selected){
                pool.execute(new ProcessEachAttributeList(col,attribute_list,class_list));
            }

            pool.shutdown();
            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            //once had scan all column, we can get the best (feature,threshold,gain) for each alive tree node
            int cur_level_node_size = alive_nodes.size();
            Queue<TreeNode> new_tree_nodes = new LinkedList<>();

            for(int i=0;i<cur_level_node_size;i++){
                //pop each alive treenode
                TreeNode treenode = alive_nodes.poll();
                double[] ret = treenode.get_best_feature_threshold_gain();
                double best_feature = ret[0];
                double best_threshold = ret[1];
                double best_gain = ret[2];
                double best_nan_go_to = ret[3];

                if(best_gain<=0){
                    //this node is leaf node
                    double leaf_score = calculate_leaf_score(treenode.Grad,treenode.Hess);
                    treenode.leaf_node_setter(leaf_score,true);
                }else {
                    //this node is internal node
                    TreeNode left_child = new TreeNode(3*treenode.index-1,treenode.depth+1,treenode.feature_dim,false);
                    TreeNode right_child = new TreeNode(3*treenode.index+1,treenode.depth+1,treenode.feature_dim,false);
                    TreeNode nan_child = null;
                    if(best_nan_go_to==0){
                        //this case we create the nan child
                        nan_child = new TreeNode(3*treenode.index,treenode.depth+1,treenode.feature_dim,false);
                        nan_nodes_cnt+=1;
                    }
                    treenode.internal_node_setter(best_feature,best_threshold,best_nan_go_to,nan_child,left_child,right_child,false);

                    new_tree_nodes.offer(left_child);
                    new_tree_nodes.offer(right_child);
                    if(nan_child != null){
                        new_tree_nodes.offer(nan_child);
                    }
                }

            }

            //update class_list.corresponding_tree_node
            class_list.update_corresponding_tree_node(attribute_list);

            //update (Grad,Hess,num_sample) for each new tree node
            class_list.update_Grad_Hess_numsample_for_tree_node();

            //update Grad_missing, Hess_missing for each new tree node
            class_list.update_grad_hess_missing_for_tree_node(attribute_list.missing_value_attribute_list);

            //process the new tree nodes
            //satisfy max_depth? min_child_weight? min_sample_split?
            //if yes, it is leaf node, calculate its leaf score
            //if no, put into self.alive_node
            while(new_tree_nodes.size()!=0){
                TreeNode treenode = new_tree_nodes.poll();
                if(treenode.depth>=max_depth || treenode.Hess<min_child_weight || treenode.num_sample<=min_sample_split){
                    treenode.leaf_node_setter(calculate_leaf_score(treenode.Grad,treenode.Hess),true);
                }else {
                    alive_nodes.offer(treenode);
                }
            }
        }
    }


    class PredictCallable implements Callable{
        private Double[] feature;
        public PredictCallable(Double[] feature){
            this.feature = feature;
        }
        @Override
        public Double call(){
            TreeNode cur_tree_node = root;
            while(!cur_tree_node.is_leaf){
                if(feature[cur_tree_node.split_feature]==null){
                    if(cur_tree_node.nan_go_to==0){
                        cur_tree_node = cur_tree_node.nan_child;
                    }else if(cur_tree_node.nan_go_to==1){
                        cur_tree_node = cur_tree_node.left_child;
                    }else if(cur_tree_node.nan_go_to==2){
                        cur_tree_node = cur_tree_node.right_child;
                    }else {
                        if(cur_tree_node.left_child.num_sample>cur_tree_node.right_child.num_sample){
                            cur_tree_node = cur_tree_node.left_child;
                        }else {
                            cur_tree_node = cur_tree_node.right_child;
                        }
                    }

                }else if(feature[cur_tree_node.split_feature]<=cur_tree_node.split_threshold){
                    cur_tree_node = cur_tree_node.left_child;
                }else {
                    cur_tree_node = cur_tree_node.right_child;
                }
            }
            return cur_tree_node.leaf_score;
        }
    }


    public double[] predict(Double[][] features){
        ExecutorService pool = Executors.newFixedThreadPool(num_thread);
        List<Future> list = new ArrayList<Future>();
        for(int i=0;i<features.length;i++){
            Callable c = new PredictCallable(features[i]);
            Future f = pool.submit(c);
            list.add(f);
        }

        pool.shutdown();
        try {
            pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        double[] ret = new double[features.length];
        for(int i=0;i<ret.length;i++){
            try{
                ret[i] = (double) list.get(i).get();
            }catch (InterruptedException e){
                e.printStackTrace();
            }catch (ExecutionException e){
                e.printStackTrace();
            }
        }
        return ret;
    }

    private void clean_up(){
        this.alive_nodes = null;
    }

    private double sum(double[] vals){
        double s = 0;
        for(double v:vals) s+=v;
        return s;
    }

}

