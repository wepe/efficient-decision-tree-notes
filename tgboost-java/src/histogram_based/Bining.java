//cut each feature into max_bin bins with equal distinct value
//in our implement, uint8 is used to represent the feature value after bining.
//so max value of max_bin is 256, it is good enough in practice
//this class maintain bins_upper_bounder for each feature

package histogram_based;

import main.Data;

import java.util.ArrayList;
import java.util.Map;

public class Bining {
    private int max_bin, estimation_sampling;
    public ArrayList<Map<Byte,Double>> bins_upper_bounder;

    public Bining(int max_bin, int estimation_sampling, Data data){
        this.max_bin = max_bin;
        this.estimation_sampling = estimation_sampling;
        construct_bins_upper_bounder(data);
    }

    public Bining(Data data){
        this.max_bin = 256;
        this.estimation_sampling = 100000;
        construct_bins_upper_bounder(data);
    }

    private void construct_bins_upper_bounder(Data data){

    }

    public Map<Byte,Double> get(int col){
        return bins_upper_bounder.get(col);
    }



}
