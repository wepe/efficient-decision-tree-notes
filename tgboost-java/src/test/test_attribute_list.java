package test;

import main.AttributeList;
import main.Data;

public class test_attribute_list {
    public static void main(String[] args){
        Data data = new Data("/home/wepon/PycharmProjects/train.csv");
        AttributeList atl = new AttributeList(data);
        for(int i=0;i<atl.feature_dim;i++){
            for(int j=0;j<20;j++){
                System.out.print(atl.attribute_list[i][j][0]);
                System.out.print(",");
            }
            System.out.println();
        }
    }
}
