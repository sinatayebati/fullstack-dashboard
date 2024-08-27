import express from "express";
import {
  getProducts,
  getCustomers,
  getTranactions
} from "../controllers/client.js";

const router = express.Router();

router.get("/products", getProducts);
router.get("/customers", getCustomers);
router.get("/transactions", getTranactions)

export default router;
