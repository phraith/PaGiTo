import Card from "@mui/material/Card/Card";
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import {LineChart} from 'echarts/charts';
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
import { AnyTxtRecord } from "dns";

// Register the required components
echarts.use(
  [TitleComponent, TooltipComponent, GridComponent, LineChart, CanvasRenderer]
);

interface Graph {
    data: AnyTxtRecord
}

const Graph = (props: Graph) => {

    const toTraces = (data) => {
        if (data === undefined)
        {
            return []
        }
        let traces = []
        let k = data.map((x: number, index: number) => { return [index, x] })
        traces.push(k)
        console.log("traces")
        console.log(traces)
        return k
    }

    const option = {
        title: {
            text: 'Data'
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
            data: ['Data'],
            left: 400

        },

        series: [
            {
                name: "Data",
                data: toTraces(props.data),
                showSymbol: false,
                type: 'line',
                lineStyle: {
                    color: "#4bddf0"
                }
            }
        ]
    };
    return (
        <Card sx={{height: "100%", width: "100%"}}>
            {/* <ReactEcharts theme={"dark"} option={option} /> */}
            <ReactEChartsCore style={{ height: '100%', width: '100%' }} echarts={echarts}  option={option} />
        </Card>
    );
}

export default Graph;