import React, { useState, useMemo } from 'react';
import { 
    useTheme,
    Box,
    Typography,
} from '@mui/material';
import { styled } from '@mui/system';
import Header from 'components/Header';
import { ResponsiveLine } from '@nivo/line';
import { useGetSalesQuery } from 'state/api';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";

const Daily = () => {
    // TODO: update the logic with a global date state setting 
    const [startDate, setStartDate] = useState(new Date("2021-02-01"))
    const [endDate, setEndDate] = useState(new Date("2021-03-01"))
    const { data } = useGetSalesQuery();
    const theme = useTheme();

    const [formattedData] = useMemo(() => {
        if (!data) return [];

        const { dailyData } = data;
        const totalSalesLine = {
            id: "totalSales",
            color: theme.palette.secondary.main,
            data: [],
        };
        const totalUnitsLine = {
            id: "totalUnits",
            color: theme.palette.secondary[600],
            data: [],
        };

        // formatting `dailyData` data coming from backend
        Object.values(dailyData).forEach(({ date, totalSales, totalUnits }) => {
            const dateFormatted = new Date(date);
            if (dateFormatted >= startDate && dateFormatted <= endDate) {
                const splitDate = date.substring(date.indexOf("-" + 1));

                totalSalesLine.data = [
                    ...totalSalesLine.data,
                    { x: splitDate, y: totalSales }
                ];
                totalUnitsLine.data = [
                    ...totalUnitsLine.data,
                    { x: splitDate, y: totalUnits }
                ];
            }
        });

        const formattedData = [totalSalesLine, totalUnitsLine];
        return [formattedData];
    }, [data, startDate, endDate]) // eslint-disable-line react-hooks/exhaustive-deps

    const DatePickerWrapper = styled(Box)(({ theme }) => ({
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'flex-start',
        marginLeft: theme.spacing(2),
        '& .react-datepicker-wrapper': {
          width: '100%',
        },
        '& .react-datepicker__input-container input': {
          width: '100%',
          padding: theme.spacing(1),
          border: `1px solid ${theme.palette.grey[300]}`,
          borderRadius: theme.shape.borderRadius,
          fontSize: '0.875rem',
        },
      }));
      
      const DateLabel = styled(Typography)(({ theme }) => ({
        marginBottom: theme.spacing(0.5),
        fontSize: '0.75rem',
        fontWeight: 'bold',
        color: theme.palette.text.secondary,
      }));

  return (
    <Box m="1.5rem 2.5rem">
        <Header title="DAILY SALES" subTitle="Chart of daily sales" />
        <Box height="75vh">
            <Box display="flex" justifyContent="flex-end" alignItems="flex-end">
            <DatePickerWrapper>
                <DateLabel>Start Date</DateLabel>
                <DatePicker 
                selected={startDate}
                onChange={(date) => setStartDate(date)}
                selectsStart
                startDate={startDate}
                endDate={endDate}
                />
            </DatePickerWrapper>
            <DatePickerWrapper>
                <DateLabel>End Date</DateLabel>
                <DatePicker 
                selected={endDate}
                onChange={(date) => setEndDate(date)}
                selectsEnd
                startDate={startDate}
                endDate={endDate}
                minDate={startDate}
                />
            </DatePickerWrapper>
            </Box>

            {data ? (
                <ResponsiveLine
                data={formattedData}
                theme={{
                    axis: {
                        domain: {
                            line: {
                                stroke: theme.palette.secondary[200]
                            }
                        },
                        legend: {
                            text: {
                                fill: theme.palette.secondary[200]
                            }
                        },
                        ticks: {
                            line: {
                                stroke: theme.palette.secondary[200],
                                strokeWidth: 1,
                            },
                            text: {
                                fill: theme.palette.secondary[200]
                            }
                        }
                    },
                    legends: {
                        text: {
                            fill: theme.palette.secondary[200]
                        }
                    },
                    tooltip: {
                        container: {
                            color: theme.palette.primary.main
                        }
                    }
                }}
                colors={{ datum: "color" }}
                margin={{ top: 50, right: 50, bottom: 70, left: 60 }}
                xScale={{ type: 'point' }}
                yScale={{
                    type: 'linear',
                    min: 'auto',
                    max: 'auto',
                    stacked: false,
                    reverse: false
                }}
                yFormat=" >-.2f"
                curve='catmullRom'
                axisTop={null}
                axisRight={null}
                axisBottom={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 90,
                    legend: "Month",
                    legendOffset: 60,
                    legendPosition: 'middle',
                    truncateTickAt: 0
                }}
                axisLeft={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: "Total",
                    legendOffset: -50,
                    legendPosition: 'middle',
                    truncateTickAt: 0
                }}
                enableGridX={false}
                enableGridY={false}
                pointSize={10}
                pointColor={{ theme: 'background' }}
                pointBorderWidth={2}
                pointBorderColor={{ from: 'serieColor' }}
                pointLabel="data.yFormatted"
                pointLabelYOffset={-12}
                enableTouchCrosshair={true}
                useMesh={true}
                legends={[
                    {
                        anchor: 'top-right',
                        direction: 'column',
                        justify: false,
                        translateX: 50,
                        translateY: 0,
                        itemsSpacing: 0,
                        itemDirection: 'left-to-right',
                        itemWidth: 80,
                        itemHeight: 20,
                        itemOpacity: 0.75,
                        symbolSize: 12,
                        symbolShape: 'circle',
                        symbolBorderColor: 'rgba(0, 0, 0, .5)',
                        effects: [
                            {
                                on: 'hover',
                                style: {
                                    itemBackground: 'rgba(0, 0, 0, .03)',
                                    itemOpacity: 1
                                }
                            }
                        ]
                    }
                ]}
            />
            ) : <>Loading...</>}
        </Box>  
    </Box>
  )
}

export default Daily
