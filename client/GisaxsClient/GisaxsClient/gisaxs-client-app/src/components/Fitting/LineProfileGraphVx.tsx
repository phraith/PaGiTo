import React from "react";
import { Group } from "@visx/group";
import { scaleTime, scaleLinear } from "@visx/scale";
import { AxisLeft, AxisBottom } from "@visx/axis";
import { LinePath } from "@visx/shape";
import { curveLinear } from "@visx/curve";



interface LineProfileGraphVxProps {
  data: any
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

  const xScale = scaleLinear({
    range: [0, xMax],
    domain: [0, Math.max(...props.data.map(x))]
  });

  const yScale = scaleLinear({
    range: [0, yMax],
    domain: [Math.max(...props.data.map(y)), Math.min(...props.data.map(y))]
  });

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <svg width={width} height={height}>
        <Group top={25} left={65}>
          <AxisLeft scale={yScale} numTicks={4} label="Intensity" />
          <AxisBottom
            scale={xScale}
            label="Pixel position"
            labelOffset={15}
            numTicks={5}
            top={yMax}
          />
          {/* {props.data.map((point, pointIndex) => (
            <circle
              key={pointIndex}
              r={5}
              cx={xScale(x(point))}
              cy={yScale(y(point))}
              stroke="#575757"
              fill="#575757"
              fillOpacity={0.5}
            />
          ))} */}
          <LinePath
            data={props.data}
            curve={curveLinear}
            x={d => xScale(x(d))}
            y={d => yScale(y(d))}
            stroke='#207b83b3'
            strokeWidth={1.5}
          />
        </Group>
      </svg>
    </div>
  );
};

export default LineProfileGraphVx;
