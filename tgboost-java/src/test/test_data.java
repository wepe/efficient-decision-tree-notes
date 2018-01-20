package test;

import main.Data;

import java.util.Arrays;

public class test_data {
    public static void main(String[] args){
        Data data = new Data("/home/wepon/PycharmProjects/train.csv");
        for(int i=0;i<data.dataset_size;i++){
            System.out.print(data.label[i]);
            System.out.print(",");
        }
        System.out.println();

        for(int i=0;i<data.feature_dim;i++){
            System.out.println(Arrays.toString(data.missing_index[i]));
        }

        for(int i=0;i<data.feature_dim;i++){
            System.out.print(data.feature_value_index[i][0][0]);
            System.out.print(",");
            System.out.println(data.feature_value_index[i][0][1]);
        }
    }
}
