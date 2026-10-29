// tiny IIR helpers for the numpy twin (double precision, DF-I)
void lfilt2(double b0, double b1, double b2, double a1, double a2, const double *x, double *y, int n) {
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (int i = 0; i < n; i++) {
        double v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
    }
}
void tpt_hp(double a1, double a2, double a3, const double *x, double *y, int n) {
    double ic1 = 0, ic2 = 0;
    for (int i = 0; i < n; i++) {
        double v3 = x[i] - ic2;
        double v1 = a1 * ic1 + a2 * v3;
        double v2 = ic2 + a2 * ic1 + a3 * v3;
        ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2;
        y[i] = x[i] - 1.414213562 * v1 - v2;
    }
}
