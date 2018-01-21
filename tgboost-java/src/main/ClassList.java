package main;

import java.util.ArrayList;
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

    public void update_pred(double eta){
        for(int i=0;i<dataset_size;i++){
//            pred[i] += eta * corresponding_tree_node[i].leaf_score;
        }
    }

    public void update_grad_hess(Loss loss,double scale_pos_weight){
        grad = loss.grad(pred,label);
        hess = loss.hess(pred,label);
        if(scale_pos_weight != 1.0){
            for(int i=0;i<dataset_size;i++){
                if(label[i]==1){
                    grad[i] *= scale_pos_weight;
                    hess[i] *= scale_pos_weight;
                }
            }
        }
    }

    public void sampling(ArrayList<Byte> row_mask){
        for(int i=0;i<dataset_size;i++){
            grad[i] *= row_mask.get(i);
            hess[i] *= row_mask.get(i);
        }
    }


}
