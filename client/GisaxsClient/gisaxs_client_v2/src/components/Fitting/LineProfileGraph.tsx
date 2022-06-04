import { Box } from '@mui/material'
import { LineCanvas, Serie } from '@nivo/line'
import React from 'react'

interface LineProfileGraphProps {
    plotData: Serie[]
}

const LineProfileGraph = (props: LineProfileGraphProps) => {
    console.log(props.plotData)
    return (
        <Box
            sx={{
                top: 0,
                paddingTop: 10,
                paddingRight: 5,
                paddingLeft: 10,
            }}
        >
            <LineCanvas
                width={2000}
                height={600}
                data={props.plotData}
                margin={{ top: 50, right: 160, bottom: 50, left: 60 }}

                axisTop={null}
                axisRight={null}
                axisLeft={{
                    tickSize: 4,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'y',
                    legendOffset: -41,
                    legendPosition: 'middle'
                }}
                axisBottom={{
                    tickSize: 0,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: 'x',
                    legendOffset: 36,
                    legendPosition: 'middle'
                }}
                enablePoints={false}
                enableGridX={false}
                enableGridY={false}
                xScale={{
                    type: 'linear',
                    min: 'auto',
                    max: 'auto'
                }}
                yScale={{
                    type: 'linear',
                    min: 'auto',
                    max: 'auto'
                }} />
        </Box>
    )
}

export default React.memo(LineProfileGraph)