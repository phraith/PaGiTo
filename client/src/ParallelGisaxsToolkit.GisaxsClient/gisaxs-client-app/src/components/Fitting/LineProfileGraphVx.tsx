// import { Group } from "@visx/group";
// import { scaleTime, scaleLinear } from "@visx/scale";
// import { AxisLeft, AxisBottom } from "@visx/axis";
// import { LinePath } from "@visx/shape";
// import { curveLinear } from "@visx/curve";
// import ParentSize from '@visx/responsive/lib/components/ParentSize';
// import { extent, bisector } from 'd3-array';
// import Box from "@mui/system/Box/Box";

// interface LineProfileGraphVxProps {
//   simulatedData: any
//   realData: any
// }

// const LineProfileGraphVx = (props: LineProfileGraphVxProps) => {

//   let data = props.simulatedData.concat(props.realData)

//   const verticalScale = (height) => scaleLinear({
//     range: [0, height],
//     domain: extent(data, d => d.y),
//     nice: true,
//   });

//   const horizontalScale = (width) => scaleLinear({
//     range: [0, width],
//     domain: extent(data, d => d.x),
//     nice: true,
//   });
//   const margin = { top: 30, left: 30 };
//   const width = 300
//   const height = 200
//   // // accessors
//   // const x = d => d.x;
//   // const y = d => d.y;

//   // // bounds
//   // const xMax = width - 120;
//   // const yMax = height - 80;

//   // let simMaxX = Math.max(...props.simulatedData.map(x))
//   // let realMaxX = Math.max(...props.realData.map(x))
//   // let maxX = Math.max(simMaxX, realMaxX)

//   // let simMaxY = Math.max(...props.simulatedData.map(y))
//   // let realMaxY = Math.max(...props.realData.map(y))
//   // let maxY = Math.max(simMaxY, realMaxY)


//   // let simMinY = Math.min(...props.simulatedData.map(y))
//   // let realMinY = Math.min(...props.realData.map(y))
//   // let minY = Math.min(simMinY, realMinY)

//   // const xScaleSimulated = scaleLinear({
//   //   range: [0, xMax],
//   //   domain: [0, maxX]
//   // });

//   // const yScaleSimulated = scaleLinear({
//   //   range: [0, yMax],
//   //   domain: [maxY, minY]
//   // });

//   // const xScaleReal = scaleLinear({
//   //   range: [0, xMax],
//   //   domain: [0, maxX]
//   // });

//   // const yScaleReal = scaleLinear({
//   //   range: [0, yMax],
//   //   domain: [maxY, minY]
//   // });

//   return (
//     <svg width={width} height={height}>
//       <Group left={margin.left}>
//         <AxisLeft scale={verticalScale(height - margin.top)} numTicks={4} label="Intensity" />
//         <AxisBottom top={height - margin.left} scale={horizontalScale(width - margin.left)} label="Pixel position" labelOffset={15} numTicks={5} />

//         <LinePath
//           data={props.simulatedData}
//           curve={curveLinear}
//           x={d => horizontalScale(width - margin.left)(d.x)}
//           y={d => verticalScale(height - margin.top)(d.y)}
//           stroke='#207b83b3'
//           strokeWidth={1.5}
//         />
//         <LinePath
//           data={props.realData}
//           curve={curveLinear}
//           x={d => horizontalScale(width - margin.left)(d.x)}
//           y={d => verticalScale(height - margin.top)(d.y)}
//           stroke='#be245fb3'
//           strokeWidth={1.5}
//         />
//       </Group>
//     </svg>
//   );
// };

// export default LineProfileGraphVx;
