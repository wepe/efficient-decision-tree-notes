package test;

import main.Data;
import main.GBM;

public class BinaryClassification {
    public static void main(String[] args){
        GBM tgb = new GBM();
        tgb.fit("/home/wepon/PycharmProjects/data/train_.csv",
                "/home/wepon/PycharmProjects/data/test_.csv",
                50,
                true,
                "auc",
                "logloss",
                0.01,
                300,
                7,
                1.0,
                0.8,
                0.8,
                1,
                20,
                1,
                0.1,
                4);
    }

}
