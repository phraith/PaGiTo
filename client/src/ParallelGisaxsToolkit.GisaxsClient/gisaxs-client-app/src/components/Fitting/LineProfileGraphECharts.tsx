import Card from "@mui/material/Card/Card";
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import {
    GridComponent,
    // ToolboxComponent,
    TooltipComponent,
    TitleComponent

} from 'echarts/components';
// Import renderer, note that introducing the CanvasRenderer or SVGRenderer is a required step
import {
    CanvasRenderer,
    // SVGRenderer,
} from 'echarts/renderers';

// Register the required components
echarts.use(
    [TitleComponent, TooltipComponent, GridComponent, LineChart, CanvasRenderer]
);

interface LineProfileGraphEChartsProps {
    simulatedData: any
    realData: any
}

const LineProfileGraph = (props: LineProfileGraphEChartsProps) => {
    const option = {
        title: {
            text: 'Lineprofile Real vs. Simulated',
            textStyle: {
                color: 'white'
            }
        },
        animation: true,
        lazyUpdate: true,
        xAxis: {
            type: 'value'
        },
        yAxis: {
            type: 'value'
        },
        legend: {
            data: ['Real', 'Simulated'],
            left: 400

        },

        series: [
            {
                name: "Real",
                data: props.realData,
                showSymbol: false,
                type: 'line',
                lineStyle: {
                    color: "#4bddf0"
                }
            },
            {
                name: "Simulated",
                data: props.simulatedData,
                showSymbol: false,
                type: 'line',
                lineStyle: {
                    color: "#f06f9e"
                }
            }
        ]
    };
    return (
        <Card sx={{ height: "100%", width: "100%", background: "#2d2d30" }}>
            <ReactEChartsCore style={{ height: '100%', width: '100%', font: 'white' }} echarts={echarts} option={option} />
        </Card>
    );
}

export default LineProfileGraph;