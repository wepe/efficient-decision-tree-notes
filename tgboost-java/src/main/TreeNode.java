//        about TreeNode.index, an example:
//                    1
//           2        3       4
//        5  6  7   8 9 10  11 12 13
//
//        index of the root node is 1,
//        its left child's index is 3*root.index-1,
//        its right child's index is 3*root.index+1,
//        the middle child is nan_child, its index is 3*root.index

package main;

import java.util.Arrays;

public class TreeNode {
    public int index;
    public int depth;
    public int feature_dim;
    public boolean is_leaf;
    public int num_sample;
    //the gradient/hessian sum of the samples fall into this tree node
    public double Grad;
    public double Hess;
    //for split finding, record the gradient/hessian sum of the left
    public double[] G_left;
    public double[] H_left;
    //when split finding, record the best threshold, gain, missing value's branch for each feature
    public double[] best_thresholds;
    public double[] best_gains;
    public double[] best_nan_go_to;
    public double nan_go_to;
    //some data fall into this tree node
    //gradient sum, hessian sum of those with missing value for each feature
    public double[] Grad_missing;
    public double[] Hess_missing;
    //internal node
    public int split_feature;
    public double split_threshold;
    public TreeNode nan_child;
    public TreeNode left_child;
    public TreeNode right_child;
    //leaf node
    double leaf_score;


    public TreeNode(int index,int depth,int feature_dim,boolean is_leaf){
        this.index = index;
        this.depth = depth;
        this.feature_dim = feature_dim;
        this.is_leaf = is_leaf;
        this.G_left = new double[feature_dim];
        this.H_left = new double[feature_dim];
        this.best_thresholds = new double[feature_dim];
        this.best_gains = new double[feature_dim];
        this.best_nan_go_to = new double[feature_dim];
        this.Grad_missing = new double[feature_dim];
        this.Hess_missing = new double[feature_dim];

        Arrays.fill(this.best_gains,-Double.MAX_VALUE);

    }

    public void Grad_add(double value){
        Grad += value;
    }

    public void Hess_add(double value){
        Hess += value;
    }

    public void num_sample_add(double value){
        num_sample += value;
    }

    public void Grad_setter(double value){
        Grad = value;
    }

    public void Hess_setter(double value){
        Hess = value;
    }

    public void reset_Grad_Hess_missing(){
        Arrays.fill(this.Grad_missing,0.0);
        Arrays.fill(this.Hess_missing,0.0);
    }

    public void update_best_split(int col,double threshold,double gain,double nan_go_to){
        if(gain > best_gains[col]){
            best_gains[col] = gain;
            best_thresholds[col] = threshold;
            best_nan_go_to[col] = nan_go_to;
        }
    }

    public double[] get_best_feature_threshold_gain(){
        int best_feature = 0;
        double max_gain = -Double.MAX_VALUE;
        for(int i=0;i<feature_dim;i++){
            if(best_gains[i]>max_gain){
                max_gain = best_gains[i];
                best_feature = i;
            }
        }

        return new double[]{best_feature,best_thresholds[best_feature],max_gain,best_nan_go_to[best_feature]};
    }

    public void internal_node_setter(double feature,double threshold,double nan_go_to,TreeNode nan_child,
                                     TreeNode left_child,TreeNode right_child,boolean is_leaf){
        this.split_feature = (int) feature;
        this.split_threshold = threshold;
        this.nan_go_to = nan_go_to;
        this.nan_child = nan_child;
        this.left_child = left_child;
        this.right_child = right_child;
        this.is_leaf = is_leaf;
        clean_up();
    }

    public void leaf_node_setter(double leaf_score,boolean is_leaf){
        this.is_leaf = is_leaf;
        this.leaf_score = leaf_score;
        clean_up();
    }

    private void clean_up(){
        //release memory
        best_thresholds = null;
        best_gains = null;
        best_nan_go_to = null;
        G_left = null;
        H_left = null;
    }

}
