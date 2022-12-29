import { Group } from "@visx/group";
import { scaleTime, scaleLinear } from "@visx/scale";
import { AxisLeft, AxisBottom } from "@visx/axis";
import { LinePath } from "@visx/shape";
import { curveLinear } from "@visx/curve";



interface LineProfileGraphVxProps {
  simulatedData: any
  realData: any
}

const LineProfileGraphVx = (props: LineProfileGraphVxProps) => {

  // dimensions
  const height = 500;
  const width = 800;

  // accessors
  const x = d => d.x;
  const y = d => d.y;

  // bounds
  const xMax = width - 120;
  const yMax = height - 80;

  let simMaxX = Math.max(...props.simulatedData.map(x))
  let realMaxX = Math.max(...props.realData.map(x))
  let maxX = Math.max(simMaxX, realMaxX)

  let simMaxY = Math.max(...props.simulatedData.map(y))
  let realMaxY = Math.max(...props.realData.map(y))
  let maxY = Math.max(simMaxY, realMaxY)


  let simMinY = Math.min(...props.simulatedData.map(y))
  let realMinY = Math.min(...props.realData.map(y))
  let minY = Math.min(simMinY, realMinY)

  const xScaleSimulated = scaleLinear({
    range: [0, xMax],
    domain: [0, maxX]
  });

  const yScaleSimulated = scaleLinear({
    range: [0, yMax],
    domain: [maxY, minY]
  });

  const xScaleReal = scaleLinear({
    range: [0, xMax],
    domain: [0, maxX]
  });

  const yScaleReal = scaleLinear({
    range: [0, yMax],
    domain: [maxY, minY]
  });

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <svg width={width} height={height}>
        <Group top={25} left={65}>
          <AxisLeft scale={yScaleSimulated} numTicks={4} label="Intensity" />
          <AxisBottom
            scale={xScaleSimulated}
            label="Pixel position"
            labelOffset={15}
            numTicks={5}
            top={yMax}
          />
          <LinePath
            data={props.simulatedData}
            curve={curveLinear}
            x={d => xScaleReal(x(d))}
            y={d => yScaleReal(y(d))}
            stroke='#207b83b3'
            strokeWidth={1.5}
          />
          <LinePath
            data={props.realData}
            curve={curveLinear}
            x={d => xScaleReal(x(d))}
            y={d => yScaleReal(y(d))}
            stroke='#be245fb3'
            strokeWidth={1.5}
          />
        </Group>
      </svg>
    </div>
  );
};

export default LineProfileGraphVx;
