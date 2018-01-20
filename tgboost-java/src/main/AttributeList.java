package main;

import java.util.Arrays;
import java.util.Comparator;

public class AttributeList {
    public int[][] missing_value_attribute_list;
    public double[][][] attribute_list;
    public int feature_dim;

    public AttributeList(Data data){
        missing_value_attribute_list = data.missing_index;
        feature_dim = data.feature_dim;
        attribute_list = data.feature_value_index;
        sort_attribute_list();
    }

    //pre-sort: for each feature,sort (value,index) by the value
    public void sort_attribute_list(){
        for(int i=0;i<feature_dim;i++){
            Arrays.sort(attribute_list[i], new Comparator<double[]>() {
                @Override
                public int compare(double[] a, double[] b) {
                    return Double.compare(a[0], b[0]);
                }
            });
        }
    }


}
