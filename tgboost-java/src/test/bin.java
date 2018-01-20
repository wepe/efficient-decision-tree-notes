package test;

import java.util.Arrays;

public class bin{
    public int[] a;
    public bin(int n){
        a = new int[n];
        Arrays.fill(a, 1);
    }

    public void display(){
        for(int i=0;i<10;i++){
            System.out.print(a[i]);
            System.out.print(" ");
        }
        System.out.println();
    }
}
