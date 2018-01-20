package main;

import java.util.Arrays;

public class ClassList {
    public int dataset_size;
    public double[] label;
    public TreeNode[] corresponding_tree_node;
    public double[] pred;
    public double[] grad;
    public double[] hess;

    public ClassList(Data data){
        dataset_size = data.dataset_size;
        label = new double[dataset_size];
        pred = new double[dataset_size];
        grad = new double[dataset_size];
        hess = new double[dataset_size];
    }

    public void initialize_pred(double first_round_pred){
        Arrays.fill(pred, first_round_pred);
    }
}
