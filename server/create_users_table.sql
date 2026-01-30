-- ============================================================================
-- USERS TABLE FOR PRIVMED
-- Run this in Supabase Dashboard → SQL Editor
-- Creates the users table for storing user profiles (labs, patients, admins)
-- ============================================================================

-- Create users table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  
  -- Basic info
  email TEXT NOT NULL,
  full_name TEXT,
  contact_phone TEXT,
  
  -- Role: 'patient', 'lab', 'central_admin'
  role TEXT NOT NULL DEFAULT 'patient',
  
  -- Lab-specific fields (only for role='lab')
  lab_name TEXT,
  assigned_lab_id UUID,
  
  -- Constraints
  CONSTRAINT valid_role CHECK (role IN ('patient', 'lab', 'central_admin'))
);

-- Add columns if they don't exist (for existing tables)
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS lab_name TEXT;
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS assigned_lab_id UUID;

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON public.users(role);
CREATE INDEX IF NOT EXISTS idx_users_lab_name ON public.users(lab_name);

-- Enable RLS
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Drop existing policies to recreate them
DROP POLICY IF EXISTS "Users can view own profile" ON public.users;
DROP POLICY IF EXISTS "Users can update own profile" ON public.users;
DROP POLICY IF EXISTS "Users can insert own profile" ON public.users;
DROP POLICY IF EXISTS "Service role can manage all users" ON public.users;
DROP POLICY IF EXISTS "Authenticated users can insert own profile" ON public.users;

-- Policy: Users can view their own profile
CREATE POLICY "Users can view own profile"
ON public.users
FOR SELECT
USING (auth.uid() = id);

-- Policy: Users can update their own profile
CREATE POLICY "Users can update own profile"
ON public.users
FOR UPDATE
USING (auth.uid() = id);

-- Policy: Allow inserting own profile during signup
-- This is the key policy that enables signup to work
CREATE POLICY "Authenticated users can insert own profile"
ON public.users
FOR INSERT
WITH CHECK (auth.uid() = id);

-- Policy: Admins can view all users (for admin dashboard)
DROP POLICY IF EXISTS "Admins can view all users" ON public.users;
CREATE POLICY "Admins can view all users"
ON public.users
FOR SELECT
USING (
  EXISTS (
    SELECT 1 FROM public.users u 
    WHERE u.id = auth.uid() 
    AND u.role = 'central_admin'
  )
);

-- Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at on row update
DROP TRIGGER IF EXISTS users_updated_at ON public.users;
CREATE TRIGGER users_updated_at
  BEFORE UPDATE ON public.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_updated_at();

-- ============================================================================
-- VERIFY SETUP
-- ============================================================================
SELECT 'Users table created successfully' as status;

-- Show existing users (only if table has data)
DO $$
BEGIN
  IF EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'users') THEN
    IF EXISTS (SELECT 1 FROM public.users LIMIT 1) THEN
      PERFORM * FROM (
        SELECT id, email, full_name, role, lab_name, created_at 
        FROM public.users 
        ORDER BY created_at DESC 
        LIMIT 10
      ) t;
    ELSE
      RAISE NOTICE 'Users table exists but is empty';
    END IF;
  ELSE
    RAISE NOTICE 'Users table created and ready for use';
  END IF;
END $$;
