import OverallStat from "../models/OverallStat.js";

export const getSales = async (req, res) => {
    try {
        const overallStats = await OverallStat.find();

        // NOTE: for now only sending the first object in database since
        // we only have data for 2021
        res.status(200).json(overallStats[0]);
    } catch (error) {
        res.status(404).json({ message: error.message })
    }
};
