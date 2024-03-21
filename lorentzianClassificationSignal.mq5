//------------------------------------------------------------------
#property link "juandelacruz.calvo@gmail.com"
//------------------------------------------------------------------

#include <movingaverages.mqh>

#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots 1

#property indicator_label1 "Trades"
#property indicator_type1 DRAW_COLOR_ARROW
#property indicator_color1 clrLimeGreen, clrRed
#property indicator_width1 2

#property indicator_label2 "Regime Filter"
#property indicator_type2 DRAW_ARROW
#property indicator_color2 clrBlue
#property indicator_width2 2

#property indicator_label3 "Volatility Filter"
#property indicator_type3 DRAW_ARROW
#property indicator_color3 clrOrange
#property indicator_width3 2

#property indicator_label4 "Predictions"
#property indicator_type4 DRAW_NONE

#property indicator_label5 "Last Distances"
#property indicator_type5 DRAW_NONE

//
//--- input parameters
//
// General Settings
input ENUM_APPLIED_PRICE source = PRICE_CLOSE; // Price used to calculate indicators
input int neighborsCount = 8;                  // Number of neighbors to consider
input int maxBarsBack = 2000;
input int featureCount = 5;     // Number of features to use for ML predictions
input int colorCompression = 1; // Compression factor for adjusting the intensity of the color scale

// Feature Engineering
// (There is no direct equivalent for grouping in MQL5; you can organize your variables based on comments or function grouping.)

// General Settings - Exits
input bool showDefaultExits = false; // Show Default Exits
input bool useDynamicExits = false;  // Use Dynamic Exits

// General Settings
input bool showTradeStats = true; // Displays the trade stats for a given configuration.
                                  // Useful for optimizing the settings in the Feature Engineering section.
                                  // This should NOT replace backtesting and should be used for calibration purposes only.
                                  // Early Signal Flips represent instances where the model changes signals before 4 bars elapse;
                                  // high values can indicate choppy (ranging) market conditions.

input bool useWorstCase = false; // Whether to use worst-case estimates for backtesting.
                                 // This option can be useful for creating a conservative estimate that is based on close prices only,
                                 // thus avoiding the effects of intrabar repainting. This option assumes that the user does not enter
                                 // when the signal first appears and instead waits for the bar to close as confirmation.
                                 // On larger timeframes, this can mean entering after a large move has already occurred.
                                 // Leaving this option disabled is generally better for those that use this indicator as a source of confluence
                                 // and prefer estimates that demonstrate discretionary mid-bar entries. Leaving this option enabled
                                 // may be more consistent with traditional backtesting results.

// Filters
input bool useVolatilityFilter = true; // Whether to use the volatility filter.

input bool useRegimeFilter = true; // Whether to use the regime filter. ESTO DEBE ESTAR TRUE
input bool useAdxFilter = false;    // Whether to use the ADX filter.

// Regime Filter
input double regimeFilterThreshold = -0.1; // Threshold for detecting Trending/Ranging markets.

// ADX Filter
input int adxFilterThreshold = 20; // Threshold for detecting Trending/Ranging markets.

// Feature Engineering
input int rsi1Period = 14;   // First RSI period
input int rsi1EmaPeriod = 1; // First EMA RSI period

input int wtPeriod1 = 10; // The primary parameter of feature 2.
input int wtPeriod2 = 11; // The secondary parameter of feature 2 (if applicable).
 
input int cciPeriod = 20;   // CCI Period
input int cciEmaPeriod = 1; // CCI EMA Period

input int adxPeriod = 20;   // ADX Period
input int adxEmaPeriod = 2; // ADX EMA Period

input int rsi2Period = 9;    // Second RSI period
input int rsi2EmaPeriod = 1; // Second EMA RSI period

input bool useEmaFilter = false; // Use EMA Filter

// Filters
input int emaPeriod = 200; // Period of the EMA used for the EMA Filter

input bool useSmaFilter = false; // Use SMA Filter
input int smaPeriod = 200;       // Period of the SMA used for the SMA Filter

// Nadaraya-Watson Kernel Regression Settings
input bool useKernelFilter = true;     // Trade with Kernel
input bool showKernelEstimate = true;  // Show Kernel Estimate
input bool useKernelSmoothing = false; // Enhance Kernel Smoothing
input int lookbackWindow = 8;          // The number of bars used for the estimation.
input float relativeWeighting = 8.0;   // Relative weighting of time frames.
input int regressionLevel = 25;        // Bar index on which to start regression.
input int lag = 2;                     // Lag for crossover detection.

// Display Settings
input bool showBarColors = true;        // Show Bar Colors
input bool showBarPredictions = true;   // Show Bar Prediction Values
input bool useAtrOffset = false;        // Use ATR Offset
input float barPredictionsOffset = 0.0; // Bar Prediction Offset

// Indicator handlers
int hATRPeriod1;
int hATRPeriod10;
int hSma;
int hFirstRSI;
int hSecondRSI;
int hCci;
int hAdxFeature;
int hAdx14;
int hEmaWT;
int hEmaFilter;
int hSmaFilter;

// Filter buffers
double emaFilterBuffer[];
double smaFilterBuffer[];

double recentAtrBuffer[];
double historicalAtrBuffer[];
double hlc4[];
//
//--- indicator buffers
//
double firstBufferRsi[];
double firstEmaRsi[];
double f1Rsi[];

double _historicMinWt = 1e10;
double _historicMaxWt = -1e10;
double hlc3[];
double wtEma1[];
double wtAbsMinusSource[];
double wtEma2[];
double wtCi[];
double wt1[];
double wt2;
double wt1MinusWt2[];

double f2WT[];

double _historicMinCCi = 1e10;
double _historicMaxCCi = -1e10;
double bufferCci[];
// double cciNormalised[];
double f3Cci[];

double bufferAdx[];
double adxEmaBuffer[];
double f4Adx[];

double secondBufferRsi[];
// double secondEmaRsi[];
double f5Rsi[];

double klmf[];
double absCurveSlope[];
double exponentialAverageAbsCurveSlope[];
double lastrRegimeFilterValue1 = 0;
double lastrRegimeFilterValue2 = 0;

double ohlc4[];
double trSmooth[2];
double smoothDirectionalMovementPlus[2];
double smoothnegMovement[2];

double trades[];
// double realTrades[];
double tradeColors[];
// double lastDistances[]; // Distances
// double regimeBuffer[];
// double volatilityBuffer[];
// double predictionsBuffer[];

// ML model
double predictions[];
double prediction = 0;
double distances[];
double distance = 0;
double y_train_array[];
int y_train_series;
int signal[];
int maxBarsBackIndex;
int barsHeld = 0;

bool isEmaUptrend[];
bool isEmaDowntrend[];
bool isSmaUptrend[];
bool isSmaDowntrend[];

bool isHeldFourBars;
bool isHeldLessThanFourBars;
bool isDifferentSignalType;
bool isEarlySignalFlip;
bool isBuySignal;
bool isSellSignal;
bool isLastSignalBuy;
bool isLastSignalSell;
bool isNewBuySignal;
bool isNewSellSignal;
bool isBullish;
bool isBearish;
bool startLongTrade;
bool startShortTrade;

double lastDistance = -1.0;
int y_train_array_size;
int sizeLoop;
int size;

bool volatilityFilter = false;
bool regimeFilter = false;
bool adxFilter = false;
bool filter_all = false;

int highestPeriodConfigured = 0;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- indicator buffers mapping
    SetIndexBuffer(0, trades, INDICATOR_DATA);

    SetIndexBuffer(1, tradeColors, INDICATOR_COLOR_INDEX);
    // SetIndexBuffer(2, regimeBuffer, INDICATOR_DATA);
    // SetIndexBuffer(3, volatilityBuffer, INDICATOR_DATA);
    // SetIndexBuffer(4, predictionsBuffer, INDICATOR_DATA);
    // SetIndexBuffer(5, lastDistances, INDICATOR_DATA);
    // SetIndexBuffer(6, realTrades, INDICATOR_CALCULATIONS);


    highestPeriodConfigured = MathMax(rsi1Period, MathMax(rsi2Period, MathMax(cciPeriod, MathMax(adxPeriod, MathMax(wtPeriod1, MathMax(wtPeriod2, MathMax(emaPeriod, smaPeriod)))))));
    // Do not calculate the indicator for the highest period
    PlotIndexSetInteger(0, PLOT_DRAW_BEGIN, highestPeriodConfigured);

    PlotIndexSetInteger(0, PLOT_ARROW, 116);
    //--- Set the vertical shift of arrows in pixels
    PlotIndexSetInteger(0, PLOT_ARROW_SHIFT, 5);
    //--- Set as an empty value 0
    PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, 0);

    PlotIndexSetInteger(0,               //  The number of a graphical style
                        PLOT_LINE_COLOR, //  Property identifier
                        1,               //  The index of the color, where we write the color
                        clrLightGreen);  //  A new color

    PlotIndexSetInteger(0,               //  The number of a graphical style
                        PLOT_LINE_COLOR, //  Property identifier
                        0,               //  The index of the color, where we write the color
                        clrRed);         //  A new color

    PlotIndexSetInteger(1, PLOT_SHOW_DATA, true);
    //--- indicator short name assignment
    IndicatorSetString(INDICATOR_SHORTNAME, "Lorentzian Classification");

    hATRPeriod1 = iATR(Symbol(), Period(), 1);
    hATRPeriod10 = iATR(Symbol(), Period(), 10);
    hFirstRSI = iRSI(Symbol(), Period(), rsi1Period, PRICE_CLOSE);
    hSecondRSI = iRSI(Symbol(), Period(), rsi2Period, PRICE_CLOSE);
    hCci = iCCI(Symbol(), Period(), cciPeriod, PRICE_CLOSE);
    hAdxFeature = iADX(Symbol(), Period(), adxPeriod);
    hAdx14 = iADX(Symbol(), Period(), 14);
    hEmaWT = iMA(Symbol(), Period(), wtPeriod1, 0, MODE_EMA, PRICE_TYPICAL);

    hEmaFilter = iMA(Symbol(), Period(), emaPeriod, 0, MODE_EMA, PRICE_CLOSE);
    hSmaFilter = iMA(Symbol(), Period(), smaPeriod, 0, MODE_SMA, PRICE_CLOSE);

    ArrayInitialize(wtAbsMinusSource, 0);
    ArrayInitialize(wtEma2, 0);
    ArrayInitialize(wtCi, 0);
    ArrayInitialize(wt1, 0);
    ArrayInitialize(wt1MinusWt2, 0);

    return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator de-initialization function                      |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    IndicatorRelease(hATRPeriod1);
    IndicatorRelease(hATRPeriod10);
    IndicatorRelease(hSma);
    IndicatorRelease(hFirstRSI);
    IndicatorRelease(hSecondRSI);
    IndicatorRelease(hCci);
    IndicatorRelease(hAdxFeature);
    IndicatorRelease(hAdx14);
    IndicatorRelease(hEmaWT);
    IndicatorRelease(hEmaFilter);
    IndicatorRelease(hSmaFilter);
}

int OnCalculate(const int rates_total, const int prev_calculated, const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{

    // Only calculate per new bar
    // if (prev_calculated == rates_total)
    // {
    //     return (rates_total);
    // }

    // ArrayResize(firstEmaRsi, rates_total);
    ArrayResize(f1Rsi, rates_total);
    // ArrayResize(secondEmaRsi, rates_total);
    ArrayResize(f5Rsi, rates_total);
    ArrayResize(f3Cci, rates_total);
    // ArrayResize(cciEmaBuffer, rates_total);
    ArrayResize(f4Adx, rates_total);
    ArrayResize(adxEmaBuffer, rates_total);

    ArrayResize(y_train_array, rates_total);
   // ArrayResize(lastDistances, rates_total);

    ArrayResize(hlc3, rates_total);
    ArrayResize(wtAbsMinusSource, rates_total);
    ArrayResize(wtEma2, rates_total);
    ArrayResize(wtCi, rates_total);
    ArrayResize(wt1, rates_total);
    ArrayResize(wt1MinusWt2, rates_total);
    ArrayResize(f2WT, rates_total);

    ArrayResize(ohlc4, rates_total);
    //ArrayResize(realTrades, rates_total);

    if (Bars(_Symbol, _Period) < rates_total)
        return (prev_calculated);

    if (!fillArrayFromBuffer(recentAtrBuffer, hATRPeriod1, 0, 0, MathMax(rates_total - prev_calculated, 2)))
    {
        Print("Error filling array from buffer");
    }
    if (!fillArrayFromBuffer(historicalAtrBuffer, hATRPeriod10, 0, 0, MathMax(rates_total - prev_calculated, 2)))
    {
        Print("Error filling array from buffer");
    }

    if (!fillArrayFromBuffer(emaFilterBuffer, hEmaFilter, 0, 0, MathMax(rates_total - prev_calculated, 2)))
    {
        Print("Error filling array from buffer");
    }

    if (!fillArrayFromBuffer(smaFilterBuffer, hSmaFilter, 0, 0, MathMax(rates_total - prev_calculated, 2)))
    {
        Print("Error filling array from buffer");
    }

    if (!fillArrayFromBuffer(firstBufferRsi, hFirstRSI, 0, 0, MathMax(rates_total - prev_calculated, 2)))
    {
        Print("Error filling array from buffer");
    }

    overrideFirstBuffer(firstBufferRsi, rsi1Period, prev_calculated);

    if (!fillArrayFromBuffer(secondBufferRsi, hSecondRSI, 0, 0, MathMax(rates_total - prev_calculated, 2)))
    {
        Print("Error filling array from buffer");
    }

    overrideFirstBuffer(secondBufferRsi, rsi2Period, prev_calculated);

    if (!fillArrayFromBuffer(bufferAdx, hAdxFeature, 0, 0, MathMax(rates_total - prev_calculated, 2)))
    {
        Print("Error filling array from buffer");
    }
    overrideFirstBuffer(bufferAdx, adxPeriod, prev_calculated);

    if (!fillArrayFromBuffer(bufferCci, hCci, 0, 0, MathMax(rates_total - prev_calculated, 2)))
    {
        Print("Error filling array from buffer");
    }

    overrideFirstBuffer(bufferCci, cciPeriod, prev_calculated);

    // EMA for WT
    if (!fillArrayFromBuffer(wtEma1, hEmaWT, 0, 0, MathMax(rates_total - prev_calculated, 2)))
    {
        Print("Error filling array from buffer");
    }
    // Prepare RSI features
    // TODO: Check if 0 is the correct value for the prev_calculated parameter
    // ExponentialMAOnBuffer(rates_total, prev_calculated, 0, rsi1EmaPeriod, firstBufferRsi, firstEmaRsi, prev_calculated != 0);
    rescaleArray(rates_total, prev_calculated, 0, 100, 0, 1, firstBufferRsi, f1Rsi);

    // ExponentialMAOnBuffer(rates_total, prev_calculated, 0, rsi2EmaPeriod, secondBufferRsi, secondEmaRsi, prev_calculated != 0);
    rescaleArray(rates_total, prev_calculated, 0, 100, 0, 1, secondBufferRsi, f5Rsi);

    // Prepare CCI feature
    // ExponentialMAOnBuffer(rates_total, prev_calculated, 0, cciEmaPeriod, bufferCci, cciEmaBuffer, prev_calculated != 0);

    for (int i = 0; i < ArraySize(bufferCci); i++)
    {
        _historicMinCCi = MathMin(bufferCci[i], _historicMinCCi);
        _historicMaxCCi = MathMax(bufferCci[i], _historicMaxCCi);
    }

    rescaleArray(rates_total, prev_calculated, _historicMinCCi, _historicMaxCCi, 0, 1, bufferCci, f3Cci);

    // Prepare ADX
    ExponentialMAOnBuffer(rates_total, prev_calculated, 0, adxEmaPeriod, bufferAdx, adxEmaBuffer);
    rescaleArray(rates_total, prev_calculated, 0, 100, 0, 1, adxEmaBuffer, f4Adx);
    // The last bar pending to calculate is always the current bar, on it high, low and close are the same since it at beginning of the bar
    // The last bar should probably not be used to feed the ML model

    // Only calculate last maxBarsBack
    int firstBar = MathMax(prev_calculated - 1, 0);
    if (prev_calculated == 0)
    {
        firstBar = rates_total > maxBarsBack ? rates_total - maxBarsBack - highestPeriodConfigured - 1 : 0;
    }

    for (int i = firstBar; i < rates_total - 1 && !IsStopped(); i++)
    {
        // Print("Calculating Bar on time: ", time[i]);
        hlc3[i] = (high[i] + low[i] + close[i]) / 3;
        // Prepare WT
        if (i > 0)
        {
            wtAbsMinusSource[i] = MathAbs(hlc3[i] - wtEma1[prev_calculated == 0 ? i : 0]);
            // if (wtAbsMinusSource[i] == 0)
            // {
            //     Print("para");
            // }
            if (i > 1)
            {

                wtEma2[i] = ExponentialMA(i, wtPeriod1, i == 0 ? 0 : wtEma2[i - 1], wtAbsMinusSource);
                wtCi[i] = (wtEma2[i] == 0 ? 0 : hlc3[i] - wtEma1[prev_calculated == 0 ? i : 0]) / (0.015 * wtEma2[i]);
                if (i > 2)
                {
                    wt1[i] = ExponentialMA(i, wtPeriod2, i == 0 ? 0 : wt1[i - 1], wtCi);
                    if (i > 8)
                    {
                        wt2 = SimpleMA(i, 4, wt1);
                        wt1MinusWt2[i] = wt1[i] - wt2;
                    }
                }
            }
        }

        // Check the new min and max
        _historicMinWt = MathMin(wt1MinusWt2[i], _historicMinWt);
        _historicMaxWt = MathMax(wt1MinusWt2[i], _historicMaxWt);

        f2WT[i] = _historicMaxWt - _historicMinWt == 0 ? 0 : (wt1MinusWt2[i] - _historicMinWt) / (_historicMaxWt - _historicMinWt);

        // Print(StringFormat("hlc3: %.2f, ema: %.2f, wtAbsMinusSource: %.2f, wtEma2: %.2f, wtCi: %.2f, wt1: %.2f, wt2: %.2f, wt1MinusWt2: %.2f, f2WT: %.2f,",
        //                   hlc3[i], wtEma1[prev_calculated == 0 ? i : 0], wtAbsMinusSource[i], wtEma2[i], wtCi[i], wt1[i], wt2, wt1MinusWt2[i], f2WT[i]));

        // Filters
        ohlc4[i] = (high[i] + low[i] + close[i] + open[i]) / 4;

        volatilityFilter = filter_volatility(recentAtrBuffer[prev_calculated == 0 ? i : 0], historicalAtrBuffer[prev_calculated == 0 ? i : 0], useVolatilityFilter);
        regimeFilter = i > 0 ? regime_filter(ohlc4, high, low, i, regimeFilterThreshold, useRegimeFilter) : false;
        adxFilter = filter_adx(useAdxFilter);

        // regimeBuffer[i] = regimeFilter ? low[i] - 0.1 : 0;
        // volatilityBuffer[i] = volatilityFilter ? high[i] + 0.1 : 0;

        filter_all = volatilityFilter && regimeFilter && adxFilter;
        // filter_all = true;

        // if (filter_all)
        // {
        //     Print("Filter ALL!");
        // }

        // // FeatureSeries Object: Calculated Feature Series based on Feature Variables
        // FeatureSeries featureSeries;
        ArrayResize(isEmaUptrend, ArraySize(isEmaUptrend) + 1);
        ArrayResize(isEmaDowntrend, ArraySize(isEmaDowntrend) + 1);
        ArrayResize(isSmaUptrend, ArraySize(isSmaUptrend) + 1);
        ArrayResize(isSmaDowntrend, ArraySize(isSmaDowntrend) + 1);

        isEmaUptrend[ArraySize(isEmaUptrend) - 1] = (!useEmaFilter || close[i] > emaFilterBuffer[prev_calculated == 0 ? i : 0]);
        isEmaDowntrend[ArraySize(isEmaDowntrend) - 1] = (!useEmaFilter || close[i] < emaFilterBuffer[prev_calculated == 0 ? i : 0]);
        isSmaUptrend[ArraySize(isSmaUptrend) - 1] = (!useSmaFilter || close[i] > smaFilterBuffer[prev_calculated == 0 ? i : 0]);
        isSmaDowntrend[ArraySize(isSmaDowntrend) - 1] = (!useSmaFilter || close[i] < smaFilterBuffer[prev_calculated == 0 ? i : 0]);

        // This model specializes specifically in predicting the direction of price action over the course of the next 4 bars.
        // To avoid complications with the ML model, this value is hardcoded to 4 bars but support for other training lengths may be added in the future.
        if (i > 4)
        {
            y_train_series = close[i - 4] < close[i] ? -1 : close[i - 4] > close[i] ? 1
                                                                                    : 0;
            y_train_array[i] = y_train_series;

            maxBarsBackIndex = rates_total >= maxBarsBack ? rates_total - maxBarsBack : 0;

            size = MathMin(maxBarsBack - 1, ArraySize(y_train_array) - 1);
            sizeLoop = MathMin(maxBarsBack - 1, size);

            // Size variable
            int firstElementToCheck = maxBarsBackIndex > 0 ? maxBarsBackIndex - sizeLoop : 0;

            // if (f3Cci[i] == 0) {
            //    Print("para");
            // }

            // Print(StringFormat("Open: %.2f, Close: %.2f, Features: f1: %.2f, f2: %.2f, f3: %.2f, f4: %.2f, f5: %.2f", open[i], close[i], f1Rsi[i], f2WT[i], f3Cci[i], f4Adx[i], f5Rsi[i]));

            lastDistance = -1;

            for (int j = firstElementToCheck; j < i; j++)
            {
                double d = get_lorentzian_distance(j, i, featureCount, f1Rsi, f2WT, f3Cci, f4Adx, f5Rsi);
                if (d >= lastDistance && j % 4 == 0)
                {
                    lastDistance = d;

                    // Add the distance and the prediction to the arrays at the end
                    // TODO estos son arrays no series!!!
                    ArrayResize(distances, ArraySize(distances) + 1);
                    distances[ArraySize(distances) - 1] = d;
                    ArrayResize(predictions, ArraySize(predictions) + 1);
                    predictions[ArraySize(predictions) - 1] = y_train_array[i];

                    if (ArraySize(predictions) > neighborsCount)
                    {
                        lastDistance = distances[(int)MathRound(neighborsCount * 3 / 4)];
                        ArrayRemove(distances, 0, 1);
                        ArrayRemove(predictions, 0, 1);
                    }
                }
            }

            //lastDistances[i] = lastDistance;
            prediction = MathSum(predictions);
            //predictionsBuffer[i] = prediction;

            // Print("Prediction: ", prediction);

            ArrayResize(signal, ArraySize(signal) + 1);
            if (prediction > 0 && filter_all)
            {
                signal[ArraySize(signal) - 1] = 1;
            }
            else if ((prediction < 0 && filter_all))
            {
                signal[ArraySize(signal) - 1] = -1;
            }
            else if (ArraySize(signal) > 1)
            {
                signal[ArraySize(signal) - 1] = signal[ArraySize(signal) - 2];
            }
            else
            {
                signal[ArraySize(signal) - 1] = 0;
            }
            // signal[ArraySize(signal) - 1] = prediction > 0 && filter_all ? 1 : (prediction < 0 && filter_all) ? -1
            //                                                                : ArraySize(signal) > 1            ? signal[ArraySize(signal) - 2]
            //                                                                                                   : 0;
            // TODO comprobar esto
            barsHeld = ArraySize(signal) > 1 ? (signal[ArraySize(signal) - 1] != signal[ArraySize(signal) - 2]) ? 0 : barsHeld + 1 : 0;
            isHeldFourBars = barsHeld == 4;
            isHeldLessThanFourBars = (0 < barsHeld && barsHeld < 4);
            isDifferentSignalType = ArraySize(signal) > 2 ? (signal[ArraySize(signal) - 1] != signal[ArraySize(signal) - 2]) : false;
            isEarlySignalFlip = ArraySize(signal) > 3 ? (signal[ArraySize(signal) - 1] != signal[ArraySize(signal) - 2]) && (signal[ArraySize(signal) - 2] != signal[ArraySize(signal) - 3] || signal[ArraySize(signal) - 2] != signal[ArraySize(signal) - 4]) : false;
            isBuySignal = ArraySize(signal) > 1 ? signal[ArraySize(signal) - 1] == 1 && isEmaUptrend[ArraySize(isEmaUptrend) - 1] && isSmaUptrend[ArraySize(isSmaUptrend) - 1] : false;
            isSellSignal = ArraySize(signal) > 1 ? signal[ArraySize(signal) - 1] == -1 && isEmaDowntrend[ArraySize(isEmaDowntrend) - 1] && isSmaDowntrend[ArraySize(isSmaDowntrend) - 1] : false;
            isLastSignalBuy = ArraySize(signal) > 5 ? signal[ArraySize(signal) - 5] == 1 && isEmaUptrend[ArraySize(isEmaUptrend) - 5] && isSmaUptrend[ArraySize(isSmaUptrend) - 5] : false;
            isLastSignalSell = ArraySize(signal) > 5 ? signal[ArraySize(signal) - 5] == -1 && isEmaDowntrend[ArraySize(isEmaDowntrend) - 5] && isSmaDowntrend[ArraySize(isSmaDowntrend) - 5] : false;
            isNewBuySignal = isBuySignal && isDifferentSignalType;
            isNewSellSignal = isSellSignal && isDifferentSignalType;

            isBullish = useKernelFilter ? (useKernelSmoothing ? isBullishSmooth : isBullishRate) : true;
            isBearish = useKernelFilter ? (useKernelSmoothing ? isBearishSmooth : isBearishRate) : true;
            // isBullish = true;
            // isBearish = true;

            // Entry Conditions: Booleans for ML Model Position Entries
            startLongTrade = isNewBuySignal && isBullish && isEmaUptrend[ArraySize(isEmaUptrend) - 1] && isSmaUptrend[ArraySize(isSmaUptrend) - 1];
            startShortTrade = isNewSellSignal && isBearish && isEmaDowntrend[ArraySize(isEmaDowntrend) - 1] && isSmaDowntrend[ArraySize(isSmaDowntrend) - 1];

            // if (startLongTrade || startShortTrade)
            // {
            //     Print("Hay trade");
            // }

            // tradesLong[i] = startLongTrade ? close[i] : 0;
            // tradesShort[i] = startShortTrade ? close[i] : 0;
            if (startLongTrade)
            {
                trades[i] = low[i] - 0.2;
                tradeColors[i] = 1;
                // realTrades[i]= -1;
            }
            else if (startShortTrade)
            {
                trades[i] = high[i] + 0.2;
                tradeColors[i] = 0;
                // realTrades[i]= 1;
            }
            else
            {
                trades[i] = 0;
                //  realTrades[i]= 0;
            }
        }
 
        //                                                                                           : iCustom(NULL, 0, "indicator_name", 0, 0); // Replace "indicator_name" with the actual name of your indicator
        // signal = signal != 0 ? signal[0] : signal[1];
    }

    // bool filter_all = filter.volatility && filter.regime && filter.adx;

    // // Filtered Signal: The model's prediction of future price movement direction with user-defined filters applied
    // signal = (prediction > 0 && filter_all) ? direction.long : (prediction < 0 && filter_all) ? direction.short
    //                                                                                           : iCustom(NULL, 0, "indicator_name", 0, 0); // Replace "indicator_name" with the actual name of your indicator
    // signal = signal != 0 ? signal[0] : signal[1];

    // // Bar-Count Filters: Represents strict filters based on a pre-defined holding period of 4 bars
    // int barsHeld = 0;
    // barsHeld = (signal[0] != signal[1]) ? 0 : barsHeld + 1;
    // bool isHeldFourBars = (barsHeld == 4);
    // bool isHeldLessThanFourBars = (0 < barsHeld && barsHeld < 4);

    // bool isDifferentSignalType = (signal != signal[1]);
    // bool isEarlySignalFlip = (signal != signal[1]) && (signal[1] != signal[2] || signal[1] != signal[3]);
    // bool isBuySignal = (signal == direction.long)&&isEmaUptrend && isSmaUptrend;
    // bool isSellSignal = (signal == direction.short)&&isEmaDowntrend && isSmaDowntrend;
    // bool isLastSignalBuy = (signal[4] == direction.long)&&isEmaUptrend[4] && isSmaUptrend[4];
    // bool isLastSignalSell = (signal[4] == direction.short)&&isEmaDowntrend[4] && isSmaDowntrend[4];
    // bool isNewBuySignal = isBuySignal && isDifferentSignalType;
    // bool isNewSellSignal = isSellSignal && isDifferentSignalType;

    // // Assuming 'rationalQuadratic' and 'gaussian' functions are defined
    // double yhat1 = rationalQuadratic(settings.source, h, r, x);
    // double yhat2 = gaussian(settings.source, h - lag, x);

    // double kernelEstimate = yhat1;

    // // Kernel Rates of Change
    // bool wasBearishRate = yhat1[2] > yhat1[1];
    // bool wasBullishRate = yhat1[2] < yhat1[1];
    // bool isBearishRate = yhat1[1] > yhat1[0];
    // bool isBullishRate = yhat1[1] < yhat1[0];
    // bool isBearishChange = isBearishRate && wasBullishRate;
    // bool isBullishChange = isBullishRate && wasBearishRate;

    // bool isBullishCrossAlert = yhat2[1] < yhat1[1] && yhat2 > yhat1;
    // bool isBearishCrossAlert = yhat2[1] > yhat1[1] && yhat2 < yhat1;
    // bool isBullishSmooth = yhat2 >= yhat1;
    // bool isBearishSmooth = yhat2 <= yhat1;

    // bool alertBullish = useKernelSmoothing ? isBullishCrossAlert : isBullishChange;
    // bool alertBearish = useKernelSmoothing ? isBearishCrossAlert : isBearishChange;

    // // Entry Conditions: Booleans for ML Model Position Entries
    // bool startLongTrade = isNewBuySignal && isBullish && isEmaUptrend && isSmaUptrend;
    // bool startShortTrade = isNewSellSignal && isBearish && isEmaDowntrend && isSmaDowntrend;

    // // TODO translate BarSinze
    // // Dynamic Exit Conditions: Booleans for ML Model Position Exits based on Fractal Filters and Kernel Regression Filters
    // bool lastSignalWasBullish = ta.Barssince(startLongTrade) < ta.Barssince(startShortTrade);
    // bool lastSignalWasBearish = ta.Barssince(startShortTrade) < ta.Barssince(startLongTrade);
    // int barsSinceRedEntry = ta.Barssince(startShortTrade);
    // int barsSinceRedExit = ta.Barssince(alertBullish);
    // int barsSinceGreenEntry = ta.Barssince(startLongTrade);
    // int barsSinceGreenExit = ta.Barssince(alertBearish);
    // bool isValidShortExit = barsSinceRedExit > barsSinceRedEntry;
    // bool isValidLongExit = barsSinceGreenExit > barsSinceGreenEntry;
    // bool endLongTradeDynamic = (isBearishChange && isValidLongExit);
    // bool endShortTradeDynamic = (isBullishChange && isValidShortExit);

    // // End Trade Conditions
    // bool endLongTradeStrict = ((isHeldFourBars && isLastSignalBuy) || (isHeldLessThanFourBars && isNewSellSignal && isLastSignalBuy)) && startLongTrade[4];
    // bool endShortTradeStrict = ((isHeldFourBars && isLastSignalSell) || (isHeldLessThanFourBars && isNewBuySignal && isLastSignalSell)) && startShortTrade[4];

    // bool isDynamicExitValid = !useEmaFilter && !useSmaFilter && !useKernelSmoothing;

    // bool endLongTrade = settings.useDynamicExits && isDynamicExitValid ? endLongTradeDynamic : endLongTradeStrict;
    // bool endShortTrade = settings.useDynamicExits && isDynamicExitValid ? endShortTradeDynamic : endShortTradeStrict;

    // New code

    // return result;
    return (rates_total);
}

double zeroIfNotAvailable(double &value[], int index)
{
    return index >= 0 && index < ArraySize(value) ? value[index] : 0;
}

double MathSum(const double &array[])
{
    int sizeV = ArraySize(array);
    if (sizeV == 0)
        return (0);
    //--- calculate sum
    double sum = 0.0;
    for (int i = 0; i < sizeV; i++)
        sum += array[i];
    //--- return sum
    return (sum);
}

bool filter_adx(bool useAdxFilterV)
{
    if (!useAdxFilterV)
    {
        return true;
    }

    double adx[1];

    if (!CopyBuffer(hAdx14, 0, 0, 1, adx))
    {
        Print("Error copying buffer");
    }

    return adx[0] > adxFilterThreshold;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void overrideFirstBuffer(double &buffer[], int period, int prev_calculated)
{
    if (prev_calculated == 0)
    {
        // Override the first values on teh buffer with the first calculated value
        for (int i = 0; i < period; i++)
        {
            buffer[i] = buffer[period];
        }
    }
}

bool crossAbove(double &indicatorA[], double &indicatorB[])
{
    return indicatorA[0] <= indicatorB[0] && indicatorA[1] > indicatorB[1];
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool crossBelow(double &indicatorA[], double &indicatorB[])
{
    return indicatorA[0] >= indicatorB[0] && indicatorA[1] < indicatorB[1];
}

double filter_volatility(double recentAtrV, double historicalAtrV, bool useVolatilityFilterV = false)
{
    if (!useVolatilityFilterV)
    {
        return true;
    }

    return recentAtrV > historicalAtrV;
}

bool regime_filter(double &src[], const double &high[], const double &low[], int currentIndex, const double threshold, const bool useRegimeFilterV)
{
    if (!useRegimeFilterV)
    {
        return true;
    }

    // Calculate the slope of the curve.
    double value1 = 0.2 * (src[currentIndex] - src[currentIndex - 1]) + 0.8 * lastrRegimeFilterValue1;
    double value2 = 0.1 * (high[currentIndex] - low[currentIndex]) + 0.8 * lastrRegimeFilterValue2;

    lastrRegimeFilterValue1 = value1;
    lastrRegimeFilterValue2 = value2;

    double omega = MathAbs(value1 / value2);
    double alpha = (-MathPow(omega, 2) + MathSqrt(MathPow(omega, 4) + 16 * MathPow(omega, 2))) / 8;

    ArrayResize(klmf, ArraySize(src));
    klmf[currentIndex] = (alpha * src[currentIndex]) + ((1 - alpha) * zeroIfNotAvailable(klmf, currentIndex - 1));

    ArrayResize(absCurveSlope, ArraySize(src));
    ArrayResize(exponentialAverageAbsCurveSlope, ArraySize(src));
    absCurveSlope[currentIndex] = MathAbs(klmf[currentIndex] - zeroIfNotAvailable(klmf, currentIndex - 1));
    exponentialAverageAbsCurveSlope[currentIndex] = 1.0 * ExponentialMA(currentIndex, 200, zeroIfNotAvailable(exponentialAverageAbsCurveSlope, currentIndex - 1), absCurveSlope);
    double normalized_slope_decline = exponentialAverageAbsCurveSlope[currentIndex] == 0 ? 0 : (absCurveSlope[currentIndex] - exponentialAverageAbsCurveSlope[currentIndex]) / exponentialAverageAbsCurveSlope[currentIndex];
    // Calculate the slope of the curve.
    return normalized_slope_decline >= threshold;
}

//+------------------------------------------------------------------+
//|  Rescale                     |
//+------------------------------------------------------------------+
int rescaleArray(const int rates_total,
                 const int prev_calculated,
                 const double oldMin,
                 const double oldMax,
                 const double newMin,
                 const double newMax,
                 const double &price[],
                 double &buffer[])
{
    int i, j;
    double maxDifference = MathMax(oldMax - oldMin, 10e-10);
    //--- main loop
    for (i = prev_calculated, j = 0; i < rates_total && j < rates_total - prev_calculated && !IsStopped(); i++, j++)
        buffer[i] = newMin + (newMax - newMin) * (price[prev_calculated == 0 ? i : j] - oldMin) / maxDifference;

    return (rates_total);
}

//+------------------------------------------------------------------+
//|  Normalise Array                                                 |
//+------------------------------------------------------------------+
int normalizeArray(const int rates_total,
                   const int prev_calculated,
                   const double &src[],
                   const double &normalised[],
                   double &buffer[])
{

    double _historicMin = 1e10;
    double _historicMax = -1e10;

    for (int i = 0; i < ArraySize(normalised); i++)
    {
        _historicMin = MathMin(normalised[i], _historicMin);
        _historicMax = MathMax(normalised[i], _historicMax);
    }

    return rescaleArray(rates_total, prev_calculated, _historicMin, _historicMax, 0, 1, src, buffer);
}

double get_lorentzian_distance(int i, int latestFeature, int featureCountV, double &f1RsiV[], double &f2WTV[], double &f3CciV[], double &f4AdxV[], double &f5RsiV[])
{
    double result = 0.0;

    // if (ArraySize(f2WTV) == i)
    // {
    //     Print("para");
    // }

    switch (featureCountV)
    {
    case 5:
        return MathLog(1 + MathAbs(f1RsiV[latestFeature] - f1RsiV[i])) +
               MathLog(1 + MathAbs(f2WTV[latestFeature] - f2WTV[i])) +
               MathLog(1 + MathAbs(f3CciV[latestFeature] - f3CciV[i])) +
               MathLog(1 + MathAbs(f4AdxV[latestFeature] - f4AdxV[i])) +
               MathLog(1 + MathAbs(f5RsiV[latestFeature] - f5RsiV[i]));
        break;
    case 4:
        return MathLog(1 + MathAbs(f1RsiV[latestFeature] - f1RsiV[i])) +
               MathLog(1 + MathAbs(f2WTV[latestFeature] - f2WTV[i])) +
               MathLog(1 + MathAbs(f3CciV[latestFeature] - f3CciV[i])) +
               MathLog(1 + MathAbs(f4AdxV[latestFeature] - f4AdxV[i]));
        break;
    case 3:
        return MathLog(1 + MathAbs(f1RsiV[latestFeature] - f1RsiV[i])) +
               MathLog(1 + MathAbs(f2WTV[latestFeature] - f2WTV[i])) +
               MathLog(1 + MathAbs(f3CciV[latestFeature] - f3CciV[i]));
        break;
    case 2:
        return MathLog(1 + MathAbs(f1RsiV[latestFeature] - f1RsiV[i])) +
               MathLog(1 + MathAbs(f2WTV[latestFeature] - f2WTV[i]));
        break;
    }

    return result;
}

double rationalQuadratic(const double &_src[], const int _lookback, const double _relativeWeight, const int startAtBar)
{
    double _currentWeight = 0.0;
    double _cumulativeWeight = 0.0;
    int _size = ArraySize(_src);

    for (int i = 0; i < _size + startAtBar; i++)
    {
        double y = _src[i];
        double w = MathPow(1 + (MathPow(i, 2) / ((MathPow(_lookback, 2) * 2 * _relativeWeight))), -_relativeWeight);
        _currentWeight += y * w;
        _cumulativeWeight += w;
    }

    double yhat = _cumulativeWeight != 0.0 ? _currentWeight / _cumulativeWeight : 0.0;
    return yhat;
}

void gaussian(const double &src[], double &result[], const int &lookback, const int &startAtBar)
{

    for (int i = 0; i < ArraySize(src); i++)
    {
        double currentWeight = 0.0;
        double cumulativeWeight = 0.0;

        for (int j = 0; j < ArraySize(src); j++)
        {
            double y = src[j];
            double w = MathExp(-MathPow(j, 2) / (2 * MathPow(lookback, 2)));
            currentWeight += y * w;
            cumulativeWeight += w;
        }

        double yhat = currentWeight / cumulativeWeight;
        ArrayResize(result, i + 1);
        result[i] = yhat;
    }
}

bool fillArrayFromBuffer(double &buffer[],     // indicator buffer
                         int ind_handle,       // handle of the indicator
                         int indicator_buffer, // no buffer of the indicator buffer
                         int start_pos,        // start position in the indicator buffer
                         int count             // number of copied values
)
{
    //--- reset error code
    ResetLastError();
    //--- fill a part of the iRSIBuffer array with values from the indicator buffer that has 0 index
    if (CopyBuffer(ind_handle, indicator_buffer, start_pos, count, buffer) < 0)
    {
        //--- if the copying fails, tell the error code
        PrintFormat("Failed to copy data from indicator, error code %d", GetLastError());
        //--- quit with zero result - it means that the indicator is considered as not calculated
        return (false);
    }
    //--- everything is fine
    return (true);
}
