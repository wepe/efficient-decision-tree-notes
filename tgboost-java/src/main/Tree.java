package main;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

public class Tree {
    public TreeNode root;
    public int min_sample_split;
    public int min_child_weight;
    public int max_depth;
    public double colsample;
    public double rowsample;
    public double lambda;
    public double gamma;
    public int num_thread;
    public Queue<TreeNode> alive_nodes = new LinkedList<>();
    public HashMap<Integer,TreeNode> index_to_node = new HashMap<>();
    //number of tree node of this tree
    public int nodes_cnt = 0;
    //number of nan tree node of this tree
    public int nan_nodes_cnt = 0;


    public Tree(int min_sample_split,
                int min_child_weight,
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

}
